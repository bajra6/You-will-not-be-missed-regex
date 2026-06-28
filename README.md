# If you like regex, you're a psycho.
So how about a Transformer to help humankind with it?

A Transformer built from scratch that converts English text prompts into regular expressions.

Trained on 10K natural language → regex pairs, achieves **95.9% exact match** and **99.9% valid regex rate**.
A Synthetic dataset was used. 

## Architecture

All components built from scratch in PyTorch — no pretrained models, no HuggingFace Transformers.

```
Input: "find a string of two uppercase letters followed by one digit"
    │
    ▼
BPE Tokenizer (vocab=4000)
    │
    ▼
Token Embedding (256-dim) + Sinusoidal Positional Encoding
    │
    ▼
Encoder (4× EncoderBlock)
    ├── Multi-Head Self-Attention (4 heads, 64 dim/head)
    ├── Add & LayerNorm
    ├── Positionwise Feed-Forward (1024 → ReLU → 256)
    └── Add & LayerNorm
    │
    ▼
Decoder (4× DecoderBlock)
    ├── Masked Multi-Head Self-Attention
    ├── Add & LayerNorm
    ├── Cross-Attention (queries from decoder, keys/values from encoder)
    ├── Add & LayerNorm
    ├── Positionwise Feed-Forward
    └── Add & LayerNorm
    │
    ▼
Linear Projection (256 → 256 vocab)
    │
    ▼
BPE Detokenizer (vocab=256)
    │
    ▼
Output: "^[A-Z]{2}[0-9]{1}$"
```

### Components

| Component | Description |
|-----------|-------------|
| **Multi-Head Attention** | Scaled dot-product: Q·K^T / √d_k with 4 parallel heads |
| **Positionwise FFN** | 2-layer network: Linear(256→1024) → ReLU → Linear(1024→256) |
| **LayerNorm** | Learnable γ and β, mean/std normalization along last dim |
| **Sinusoidal PE** | Fixed sin/cos waves at different frequencies — no learned params |
| **Post-LayerNorm** | Residual → Add → Norm (original paper style) |

## Hyperparameters

| Param | Value | Why |
|-------|-------|-----|
| `d_model` | 256 | Fits 4GB VRAM, expressive enough |
| `n_heads` | 4 | 64 dim/head (matches original paper ratio) |
| `d_ff` | 1024 | 4× d_model (standard) |
| `n_layers` | 4 | Deep enough without overfitting 10K samples |
| `dropout` | 0.2 | Regularization for small dataset |
| `batch_size` | 32 | Fits VRAM comfortably |
| `vocab_nl` | 4000 | BPE on descriptions |
| `vocab_regex` | 256 | BPE on regex strings |
| `warmup_steps` | 4000 | Linear warmup then 1/√step decay |
| `label_smoothing` | 0.1 | Soft targets for better generalization |
| `Total params` | 8.5M | |

## Results

| Metric | Score |
|--------|-------|
| Exact match | **95.90%** (959/1000) |
| Valid regexes | **99.90%** (999/1000) |
| Avg BLEU | **0.9376** |

## Project Structure

```
dataset/
  nl2rx.csv                    10K prompt → regex pairs
data/
  train.json                   Training split (8000)
  val.json                     Validation split (1000)
  test.json                    Test split (1000)
  nl_tokenizer.json            BPE tokenizer for NL
  regex_tokenizer.json         BPE tokenizer for regexes
src/
  [0] config.py                Hyperparameters and file paths
  [1] data.py                  CSV loading, normalization, splitting, Dataset, DataLoader
  [2] tokenizers.py            BPE tokenizer training and loading
  [3] model.py                 All Transformer components from scratch
  [4] train.py                 Training loop, warmup scheduler, checkpointing
  [5] infer.py                 Greedy decoding, detokenization, regex validation
  [6] evaluate.py              Exact match, validity rate, BLEU score
  [7] main.py                  Pipeline orchestrator
checkpoints/
  best_model.pt                Trained model weights
```

## How to Run

### Quick start

```bash
py -m src.main all
```

This runs: data splitting → tokenizer training → model building → training.

### Step-by-step

Each file can be run independently:

```bash
# Step 1: Load CSV, normalize, split into train/val/test
py -m src.data

# Step 2: Train BPE tokenizers
py -m src.tokenizers

# Step 3: Verify model builds
py -m src.model

# Step 4: Train the Transformer
py -m src.train

# Step 5: Run inference on sample prompts
py -m src.infer

# Step 6: Evaluate on test set
py -m src.evaluate
```

### Selective pipeline

```bash
py -m src.main data        # only data splitting
py -m src.main tokenize    # only tokenizer training
py -m src.main model       # only model verification
py -m src.main train       # only training
```

## Tools

- **PyTorch** — neural network framework
- **HuggingFace tokenizers** — BPE tokenizer training
- **NumPy / tqdm** — data handling and progress bars
