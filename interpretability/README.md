## How to generate Captum attribution visualizations

### Pipeline overview

```
captum_attribution_update.py   →   attrs_*/   →   captum_visualization.py   →   visualizations_*/
     (compute scores)               (.npy)           (render HTML)                   (.html)
```

---

## Step 1 — Compute attribution scores (`captum_attribution_update.py`)

`captum_attribution_update.py` loads the fine-tuned LLaMA model with a LoRA adapter and runs one of three Captum attribution methods over a clinical note, selected via `--method`. It produces two output arrays:

- `sq_attr_raw.npy` — sequence-level attribution scores, shape `(n_input_tokens,)` — used by the visualization script
- `token_attr_raw.npy` — per-output-token attribution scores

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `meta-llama/Meta-Llama-3-8B` | Base model; see choices in the script |
| `--use_lora` | `True` | Whether to apply the LoRA adapter |
| `--lora_name` | `./model/marfan/llama3_8b_genrev_aora_raw_large.pt` | Path to the LoRA checkpoint |
| `--device` | `cuda` | Device to run on |
| `--method` | `perturbation` | Attribution method: `perturbation`, `ig`, or `gxa` |
| `--n_steps` | `50` | Approximation steps for Integrated Gradients (ignored otherwise) |
| `--note` | *(built-in default)* | Path to a plain-text clinical note file |
| `--target` | *(built-in default)* | Path to a plain-text file containing the target output string |
| `--output_dir` | `.` | Directory to save the output `.npy` files |

### Attribution methods

| `--method` | Captum class | Output folder | Notes |
|---|---|---|---|
| `perturbation` | `FeatureAblation` + `LLMAttribution` | `attrs_perturbation` | Masks tokens one at a time; slowest but model-agnostic; compatible with 8-bit quantization |
| `ig` | `LayerIntegratedGradients` + `LLMGradientAttribution` | `attrs_ig_10steps` or `attrs_ig_50steps` | Use `--n_steps 10` or `--n_steps 50`; requires float16 |
| `gxa` | `LayerGradientXActivation` + `LLMGradientAttribution` | `attrs_gxa` | Single-pass gradient method; fastest; requires float16 |

### Example usage

```bash
# Perturbation (FeatureAblation)
python captum_attribution_update.py \
    --method     perturbation \
    --note       ./interpretability/notes/patient_001.txt \
    --target     ./interpretability/notes/patient_001_target.txt \
    --output_dir ./interpretability/attrs_perturbation

# Integrated Gradients — 50 steps
python captum_attribution_update.py \
    --method     ig \
    --n_steps    50 \
    --note       ./interpretability/notes/patient_001.txt \
    --target     ./interpretability/notes/patient_001_target.txt \
    --output_dir ./interpretability/attrs_ig_50steps

# Integrated Gradients — 10 steps
python captum_attribution_update.py \
    --method     ig \
    --n_steps    10 \
    --note       ./interpretability/notes/patient_001.txt \
    --target     ./interpretability/notes/patient_001_target.txt \
    --output_dir ./interpretability/attrs_ig_10steps

# Gradient × Activation
python captum_attribution_update.py \
    --method     gxa \
    --note       ./interpretability/notes/patient_001.txt \
    --target     ./interpretability/notes/patient_001_target.txt \
    --output_dir ./interpretability/attrs_gxa
```

After each run, rename `sq_attr_raw.npy` inside `--output_dir` to `<note_name>_sq_attr.npy` so the visualization script can find it.

---

## Step 2 — Render visualizations (`captum_visualization.py`)

### Prerequisites

The script expects the following directory layout:

```
interpretability/
├── notes/                    # plain-text clinical notes (.txt files)
├── keywords/                 # per-note keyword lists (.txt files, Python list literal)
├── attrs_gxa/                # Gradient × Activation attributions
├── attrs_ig_10steps/         # Integrated Gradients (10 steps)
├── attrs_ig_50steps/         # Integrated Gradients (50 steps)
├── attrs_perturbation/       # Feature Ablation (perturbation-based)
├── visualizations_manuscripts_gxa/
├── visualizations_manuscripts_ig_10steps/
├── visualizations_manuscripts_ig_50steps/
├── visualizations_manuscripts_perturbation/
└── captum_visualization.py
```

Each `attrs_*/` folder contains one `<name>_sq_attr.npy` file per note. The matching `visualizations_manuscripts_*/` folder receives the HTML output.

Each file in `keywords/` must be a Python list literal of medical terms, e.g.:
```
['aortic root dilation', 'ectopia lentis', 'Marfan syndrome']
```

### Usage

Point `--attr_path` and `--output_path` at the corresponding folder pair for the method you want to visualize:

```bash
# Gradient × Activation
python captum_visualization.py \
    --attr_path   ./attrs_gxa \
    --output_path ./visualizations_gxa

# Integrated Gradients (50 steps)
python captum_visualization.py \
    --attr_path   ./attrs_ig_50steps \
    --output_path ./visualizations_ig_50steps

# Integrated Gradients (10 steps)
python captum_visualization.py \
    --attr_path   ./attrs_ig_10steps \
    --output_path ./visualizations_ig_10steps

# Feature Ablation (perturbation-based)
python captum_visualization.py \
    --attr_path   ./attrs_perturbation \
    --output_path ./visualizations_perturbation
```

To process a single note instead of the whole directory, pass its base name (without `.txt`):

```bash
python captum_visualization.py \
    --attr_path   ./attrs_gxa \
    --output_path ./visualizations_gxa \
    --file_name   case1 \
    --temperature 8
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--note_path` | `./notes` | Directory containing clinical note `.txt` files |
| `--attr_path` | `./attrs_gxa` | Directory containing `<name>_sq_attr.npy` attribution files |
| `--output_path` | `./visualizations_nsp_final_gxa` | Directory where HTML visualizations are written |
| `--file_name` | *(all notes)* | Base name of a single note to process (omit `.txt`) |
| `--temperature` | `15` | Softmax temperature for attribution shading (see below) |
| `--tokenizer` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace tokenizer to use |

### Tuning `--temperature`

Attribution scores are passed through a softmax scaled by `temperature` before mapping to highlight intensity:

```
weight = exp(score / temperature)
```

- **Lower temperature** (e.g. `5–8`) — sharpens contrast; only the highest-scoring tokens get strong highlighting, making the most important keywords stand out clearly.
- **Higher temperature** (e.g. `15–30`) — spreads highlighting more evenly across tokens; useful for exploratory review when you don't want to miss lower-ranked terms.

A good starting point is `--temperature 8` for a focused view and `--temperature 15` (default) for a broader overview.

### Output

One `.html` file per note is written to `--output_path`. Each token in the clinical note is highlighted with a blue shade proportional to its attribution score, restricted to tokens that fuzzy-match a medical keyword from the corresponding keyword file.
