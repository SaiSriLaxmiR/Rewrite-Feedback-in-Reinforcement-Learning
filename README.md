# Incorporating Rewrite Feedback in RLHF

## Overview

Standard RLHF trains reward models on binary preference labels: annotators say "Response A is better than Response B" with no explanation of *why*. This project investigates whether augmenting preference pairs with **natural language rewrite feedback** — explicit instructions explaining what is wrong and how to fix it — produces a richer supervisory signal for reward model training.

## Key Result

Trained on **100 synthetic samples**, the rewrite-augmented reward model achieves a **3.9× larger score gap** between good and bad responses compared to a binary-label baseline — holding across both in-distribution and out-of-distribution test cases.

### In-Distribution Results

| Prompt | Rewrite Gap | Baseline Gap | Improvement |
|--------|-------------|--------------|-------------|
| Explain gradient descent | 0.4084 | 0.2235 | +83% |
| Precision vs Recall | 1.3984 | 0.1386 | +909% |
| BERT vs GPT | 0.4908 | 0.1971 | +149% |

### Out-of-Distribution Results

| Prompt | Rewrite Gap | Baseline Gap | Improvement |
|--------|-------------|--------------|-------------|
| How a refrigerator works | 0.2290 | 0.0590 | +288% |
| Capital of Australia | 0.1473 | 0.0707 | +108% |
| Haiku about ML | -0.0601 | 0.0149 | ✗ (low domain coverage) |

**Rewrite model wins 5/6 cases. Average gap 3.9× larger than baseline.**

> ⚠️ Preliminary results on 100 samples. Scaling to 500+ samples is in progress.

## How It Works

Each training sample contains 5 fields instead of the standard 2:

```
Standard RLHF:   (prompt, chosen, rejected)
This project:    (prompt, chosen, rejected, rewrite_feedback, rewritten)
```

The rewrite feedback explicitly states what is wrong with the rejected response and how to fix it. The rewritten response (improved version) is treated as an additional chosen sample during training — effectively doubling the training signal per prompt.

## Pipeline

```
Seed Prompts
     ↓
Groq / LLaMA-3  →  Synthetic Dataset (chosen, rejected, feedback, rewritten)
     ↓
DeBERTa-v3-small + Scalar Head
     ↓
Bradley-Terry Loss + Rewrite Augmentation
     ↓
Scalar Reward Model
```

## Project Structure

```
rlhf-rewrite-feedback/
├── synthetic_data_gen.py     # Stage 1: data generation via Groq API
├── reward_model.py           # Stage 2: DeBERTa reward model training
├── colab_runner.py           # Step-by-step Colab execution script
├── rlhf_dataset.json         # 100-sample synthetic dataset
├── rlhf_dataset.csv          # Same dataset in CSV format
└── README.md
```

## Setup

```bash
pip install groq datasets transformers torch tqdm pandas scikit-learn
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

## Usage

### Step 1: Generate synthetic dataset

```python
from synthetic_data_gen import PipelineConfig, SyntheticDataPipeline

config = PipelineConfig(
    groq_api_key="YOUR_GROQ_API_KEY",
    num_samples=100,
    output_dir="./rlhf_data",
    delay_between_calls=3.0,
)
pipeline = SyntheticDataPipeline(config)
pipeline.run()
```

### Step 2: Train rewrite reward model

```python
from reward_model import RewardModelConfig, RewardModelTrainer

config = RewardModelConfig(
    data_path="./rlhf_data/rlhf_dataset.json",
    output_dir="./reward_model",
    epochs=5,
    learning_rate=1e-5,
)
trainer = RewardModelTrainer(config)
trainer.train()
```

### Step 3: Train baseline (no rewrite) for comparison

```python
# Use BaselineRewardModelTrainer from colab_runner.py
# Identical config, only difference is augment_with_rewrite=False
```

### Step 4: Compare

```python
from reward_model import score_response, RewardModelConfig
import os

rm_config = RewardModelConfig()
prompt   = "Explain gradient descent."
good_r   = "Gradient descent iteratively updates parameters by moving in the direction of steepest loss decrease, scaled by a learning rate."
bad_r    = "Gradient descent is when the model learns by going downhill somehow."

rewrite_gap  = score_response("./reward_model/best_model",          prompt, good_r, rm_config) - \
               score_response("./reward_model/best_model",          prompt, bad_r,  rm_config)
baseline_gap = score_response("./reward_model_baseline/best_model", prompt, good_r, rm_config) - \
               score_response("./reward_model_baseline/best_model", prompt, bad_r,  rm_config)

print(f"Rewrite gap:  {rewrite_gap:.4f}")
print(f"Baseline gap: {baseline_gap:.4f}")
```

## Dataset Format

Each sample in `rlhf_dataset.json`:

```json
{
  "sample_id": "sample_0001",
  "prompt": "Explain gradient descent in simple terms.",
  "response_chosen": "Detailed, accurate, well-structured response...",
  "response_rejected": "Vague, incomplete response...",
  "rewrite_feedback": "The response lacks a concrete analogy. Add the concept of a loss surface and explain the role of learning rate...",
  "rewritten_response": "Improved response after applying feedback...",
  "quality_score_chosen": 0.85,
  "quality_score_rejected": 0.25,
  "domain": "instruction_following"
}
```

## Model Architecture

```
Input: "[PROMPT] <text> [RESPONSE] <text>"
  ↓
DeBERTa-v3-small encoder (~180M params)
  ↓
[CLS] token → Dropout → Linear(768→256) → GELU → Dropout → Linear(256→1)
  ↓
Scalar reward r ∈ ℝ
```

Training loss: `L = -log(sigmoid(r_chosen - r_rejected))` (Bradley-Terry)

## Limitations

- Results are preliminary (100 samples, small validation set)
- OOD generalization is partial — creative writing domain is weak due to low training coverage

## Next Steps

- [ ] Scale to 500+ samples with balanced domain coverage
- [ ] Calibration analysis (Expected Calibration Error)
- [ ] Larger validation set for meaningful accuracy metrics
- [ ] PPO fine-tuning loop using trained reward model
- [ ] Comparison with human-annotated preference data

## References

- Ouyang et al. (2022). Training language models to follow instructions with human feedback. NeurIPS.
- Bai et al. (2022). Constitutional AI: Harmlessness from AI feedback. Anthropic.
- He et al. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. ICLR.

## Author

Ryakam Sai Sri Laxmi — IIT Bombay
