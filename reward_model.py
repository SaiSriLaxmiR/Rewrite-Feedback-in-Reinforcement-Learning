"""
Rewrite-Based Scalar Reward Model Training
===========================================
Trains a scalar reward model on the synthetic RLHF dataset.
Uses a pretrained encoder (e.g. DeBERTa or DistilBERT) with a scalar head.

Compatible with: Google Colab / Kaggle (T4 GPU)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from dataclasses import dataclass


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
@dataclass
class RewardModelConfig:
    base_model: str = "microsoft/deberta-v3-small"   # ~180MB, fits on Colab T4
    data_path: str = "./rlhf_data/rlhf_dataset.json"
    output_dir: str = "./reward_model"
    max_length: int = 512
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class RLHFPairDataset(Dataset):
    """
    Each sample: (prompt + response_chosen) vs (prompt + response_rejected)
    Label: 1 means chosen > rejected
    Also supports rewrite-augmented samples: (prompt + rewritten_response)
    treated as an additional chosen sample.
    """
    def __init__(self, samples: list, tokenizer, max_length: int, augment_with_rewrite: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []

        for s in samples:
            # Standard preference pair
            self.pairs.append({
                "chosen_text": f"[PROMPT] {s['prompt']} [RESPONSE] {s['response_chosen']}",
                "rejected_text": f"[PROMPT] {s['prompt']} [RESPONSE] {s['response_rejected']}",
            })
            # Rewrite augmentation: treat rewritten as an additional chosen
            if augment_with_rewrite and s.get("rewritten_response"):
                self.pairs.append({
                    "chosen_text": f"[PROMPT] {s['prompt']} [RESPONSE] {s['rewritten_response']}",
                    "rejected_text": f"[PROMPT] {s['prompt']} [RESPONSE] {s['response_rejected']}",
                })

    def __len__(self):
        return len(self.pairs)

    def _encode(self, text: str):
        return self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        chosen_enc = self._encode(pair["chosen_text"])
        rejected_enc = self._encode(pair["rejected_text"])
        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class ScalarRewardModel(nn.Module):
    """
    Encoder + linear scalar head.
    Outputs a single scalar reward score per (prompt, response) pair.
    Trained with Bradley-Terry preference loss.
    """
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.base_model)
        hidden_size = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        reward = self.reward_head(cls_output)
        return reward.squeeze(-1)  # (batch_size,)


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────
def bradley_terry_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry pairwise ranking loss.
    Maximizes P(chosen > rejected) = sigmoid(r_chosen - r_rejected).
    Loss = -log(sigmoid(r_chosen - r_rejected))
    """
    return -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()


def rewrite_consistency_loss(reward_rewritten: torch.Tensor,
                              reward_chosen: torch.Tensor,
                              margin: float = 0.1) -> torch.Tensor:
    """
    Encourages rewritten responses to score higher than originals.
    Optional auxiliary loss.
    """
    return torch.clamp(margin - (reward_rewritten - reward_chosen), min=0).mean()


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────
class RewardModelTrainer:
    def __init__(self, config: RewardModelConfig):
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        print(f"\nDevice: {config.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        self.model = ScalarRewardModel(config).float().to(config.device)
        os.makedirs(config.output_dir, exist_ok=True)

    def _load_data(self):
        with open(self.config.data_path) as f:
            samples = json.load(f)
        split = int(0.9 * len(samples))
        train_samples = samples[:split]
        val_samples = samples[split:]
        print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

        train_dataset = RLHFPairDataset(train_samples, self.tokenizer,
                                         self.config.max_length, augment_with_rewrite=True)
        val_dataset = RLHFPairDataset(val_samples, self.tokenizer,
                                       self.config.max_length, augment_with_rewrite=False)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                  shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size,
                                shuffle=False, num_workers=2)
        return train_loader, val_loader

    def _evaluate(self, loader) -> dict:
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in loader:
                chosen_ids = batch["chosen_input_ids"].to(self.config.device)
                chosen_mask = batch["chosen_attention_mask"].to(self.config.device)
                rejected_ids = batch["rejected_input_ids"].to(self.config.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.config.device)

                r_chosen = self.model(chosen_ids, chosen_mask)
                r_rejected = self.model(rejected_ids, rejected_mask)
                loss = bradley_terry_loss(r_chosen, r_rejected)
                total_loss += loss.item()
                correct += (r_chosen > r_rejected).sum().item()
                total += len(r_chosen)
        return {"loss": total_loss / len(loader), "accuracy": correct / total}

    def train(self):
        train_loader, val_loader = self._load_data()
        optimizer = AdamW(self.model.parameters(),
                          lr=self.config.learning_rate,
                          weight_decay=self.config.weight_decay)
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_val_acc = 0
        history = []

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

            for batch in pbar:
                chosen_ids = batch["chosen_input_ids"].to(self.config.device)
                chosen_mask = batch["chosen_attention_mask"].to(self.config.device)
                rejected_ids = batch["rejected_input_ids"].to(self.config.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.config.device)

                r_chosen = self.model(chosen_ids, chosen_mask)
                r_rejected = self.model(rejected_ids, rejected_mask)
                loss = bradley_terry_loss(r_chosen, r_rejected)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            val_metrics = self._evaluate(val_loader)
            avg_train_loss = epoch_loss / len(train_loader)
            history.append({"epoch": epoch+1, "train_loss": avg_train_loss, **val_metrics})

            print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f}")

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                self._save_model("best_model")
                print(f"  ✓ New best model saved (val_acc={best_val_acc:.4f})")

        self._save_model("final_model")
        self._save_history(history)
        print(f"\n✓ Training complete. Best val accuracy: {best_val_acc:.4f}")

    def _save_model(self, name: str):
        path = os.path.join(self.config.output_dir, name)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        self.tokenizer.save_pretrained(path)

    def _save_history(self, history: list):
        path = os.path.join(self.config.output_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(history, f, indent=2)


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def score_response(model_dir: str, prompt: str, response: str,
                   config: RewardModelConfig) -> float:
    """Load trained reward model and score a single (prompt, response) pair."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ScalarRewardModel(config)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"),
                                     map_location=config.device))
    model.to(config.device)
    model.eval()

    text = f"[PROMPT] {prompt} [RESPONSE] {response}"
    enc = tokenizer(text, max_length=config.max_length, truncation=True,
                    padding="max_length", return_tensors="pt")
    with torch.no_grad():
        score = model(enc["input_ids"].to(config.device),
                      enc["attention_mask"].to(config.device))
    return score.item()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    config = RewardModelConfig(
        data_path="./rlhf_data/rlhf_dataset.json",
        output_dir="./reward_model",
        epochs=3,
        batch_size=8,
    )
    trainer = RewardModelTrainer(config)
    trainer.train()
