## How to generate Captum attribution visualizations

### Pipeline overview

```
captum_attribution.py   →   attrs_*/   →   captum_visualization.py   →   visualizations_*/
  (compute scores)           (.npy)           (render HTML)                   (.html)
```

---

## Step 1 — Compute attribution scores (`captum_attribution.py`)

`captum_attribution.py` (in the repo root) loads the fine-tuned LLaMA model with a LoRA adapter and runs a Captum attribution method over a clinical note. It produces two output arrays:

- `sq_attr_raw.npy` — sequence-level attribution scores, shape `(n_input_tokens,)` — used by the visualization script
- `token_attr_raw.npy` — per-output-token attribution scores

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `meta-llama/Meta-Llama-3-8B` | Base model; see choices in the script |
| `--use_lora` | `True` | Whether to apply the LoRA adapter |
| `--lora_name` | `./model/marfan/llama3_8b_genrev_aora_raw_large.pt` | Path to the LoRA checkpoint |
| `--device` | `cuda` | Device to run on |

### Attribution methods

The script imports several Captum explainers. Swap the underlying class to produce each set of attribution scores:

| Folder | Captum class | Notes |
|---|---|---|
| `attrs_perturbation` | `FeatureAblation` + `LLMAttribution` | Masks input tokens one at a time; slowest but model-agnostic |
| `attrs_ig_10steps` | `LayerIntegratedGradients` + `LLMGradientAttribution` | Integrated Gradients with 10 approximation steps; fast |
| `attrs_ig_50steps` | `LayerIntegratedGradients` + `LLMGradientAttribution` | Integrated Gradients with 50 steps; slower but more accurate |
| `attrs_gxa` | Gradient × Activation + `LLMGradientAttribution` | Single-pass gradient method; fastest |

After the script finishes, copy `sq_attr_raw.npy` into the appropriate `attrs_*/` folder under `interpretability/` and rename it `<note_name>_sq_attr.npy`.

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
├── visualizations_gxa/
├── visualizations_ig_10steps/
├── visualizations_ig_50steps/
├── visualizations_perturbation/
└── captum_visualization.py
```

Each `attrs_*/` folder contains one `<name>_sq_attr.npy` file per note. The matching `visualizations_*/` folder receives the HTML output.

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
