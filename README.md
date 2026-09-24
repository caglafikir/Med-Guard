# Med-Guard

Med-Guard is a safety guardrail model for a healthcare or medical assistant. It classifies incoming user requests as **safe** or **unsafe**, and assigns each request to one of nine **intent** categories. The goal is to distinguish risky inputs, such as self-harm risk, harmful treatment requests, and manipulation, from benign general questions and chronic or acute medical situations.

## Method

The model uses a **signed-graph embedding propagation** approach built on multilingual sentence embeddings:

1. **Embedding** - Requests are encoded with `paraphrase-multilingual-mpnet-base-v2` (`sentence-transformers`).
2. **Signed graph construction** - A cosine-similarity k-NN graph is built over the training examples:
   - **Positive edges**: high-similarity examples with the same safety label (pull).
   - **Negative edges**: high-similarity, opposite-label hard negatives whose intent pair appears in the predefined `confusion_pairs` list (push).
3. **Neural network** - A shared `Linear -> ReLU -> Dropout` trunk branches into a binary safety head and a nine-class intent head.
4. **Embedding propagation** - Learned features are pulled toward similar examples through the positive graph and pushed away from conflicting examples through the negative graph.
5. **Loss function** - The total objective combines weighted BCE for safety, class-weighted CE for intent, pull loss, margin-based push loss, and propagation-consistency loss.
6. **Inductive inference** - For unseen calibration or test examples, a neighborhood is built from the closest training examples and the same propagation idea is applied inductively rather than transductively.

## Propagation equations

Let $H \in \mathbb{R}^{n \times d}$ be the learned node representations, $A_{+}$ the row-normalized positive adjacency matrix, and $A_{-}$ the row-normalized negative adjacency matrix. The propagation parameters are `alpha_pos`, `alpha_neg`, `K`, and `neg_lambda`.

### Positive propagation (pull)

The positive graph repeatedly mixes each representation with messages from same-label neighbors while retaining part of the original representation:

$$
H_{+}^{(0)} = H_0 = H
$$

$$
H_{+}^{(k+1)} = (1 - \alpha_{+})H_0 + \alpha_{+}A_{+}H_{+}^{(k)},
\qquad k = 0, \ldots, K-1
$$

After the iterative updates, one final positive message is applied:

$$
H_{+} = (1 - \alpha_{+})H_0 + \alpha_{+}A_{+}H_{+}^{(K)}
$$

Here, a larger $\alpha_{+}$ gives neighbors more influence, while the residual term preserves the original node features.

### Negative propagation (push)

The negative graph produces a message from conflicting neighbors. Its difference from the positively propagated representation is scaled to form a repulsive correction:

$$
M_{-} = A_{-}H_{+}
$$

$$
\Delta_{-} = \alpha_{-}(M_{-} - H_{+})
$$

The final representation subtracts this correction, controlled by `neg_lambda`:

$$
H_{\text{final}} = H_{+} - \lambda_{-}\Delta_{-}
$$

In the implementation, the negative correction is computed without gradient tracking, so propagation changes the representation used for downstream prediction without introducing a separate gradient path through the negative message.

### Inductive propagation

For an unseen query representation $h_q$, let $\mathcal{N}(q)$ be its training neighbors and let $w_{qi}$ be their normalized similarity weights. The inductive update is:

$$
m_{+}(q) = \sum_{i \in \mathcal{N}(q)} w_{qi}h_{+,i},
\qquad
\delta_{-}(q) = \sum_{i \in \mathcal{N}(q)} w_{qi}\Delta_{-,i}
$$

$$
h_{q,\text{final}} = (1 - \alpha_{+})h_q + \alpha_{+}m_{+}(q) - \lambda_{-}\alpha_{-}\delta_{-}(q)
$$

If no training neighbor passes the similarity threshold, the query representation is left unchanged.

## Intent categories

| Class | Description |
|---|---|
| `benign_general` | General, harmless request |
| `medication_related` | Medication-related request |
| `mild_emotional_stress` | Mild emotional distress |
| `chronic_condition` | Request related to a chronic condition |
| `non_acute_injury` | Non-acute injury |
| `acute_medical_risk` | Acute medical risk |
| `harmful_treatment` | Harmful treatment request |
| `manipulation` | Manipulative request |
| `self_harm_risk` | Self-harm risk |

## Project structure

```
Med-Guard/
├── data/
│   └── dataset_shuffled.csv  # Dataset
├── notebooks/
│   └── guard.ipynb       # Thin orchestration layer for analysis
├── src/
│   ├── config.py         # Hyperparameters and label/intent mappings
│   ├── data.py            # Data loading, embeddings, and train/calibration/test splits
│   ├── graph.py           # Signed graph construction
│   ├── propagation.py     # Transductive and inductive embedding propagation
│   ├── model.py            # SignedGraphSafetyModel architecture
│   ├── losses.py           # Pull, push, and propagation-consistency losses
│   ├── train.py             # Training loop
│   └── evaluate.py          # Inference and evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

Reusable project logic lives in modules under `src/`. The `notebooks/guard.ipynb` notebook imports those modules and presents metrics, propagation effects, and false-negative analysis step by step.

## Installation

```bash
git clone https://github.com/caglafikir/Med-Guard.git
cd Med-Guard
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS / Linux
pip install -r requirements.txt
```

The dataset is located at `data/dataset_shuffled.csv` and contains at least the following columns:

| Column | Description |
|---|---|
| `request` | User request text |
| `safety_level` | `safe` / `unsafe` |
| `intent` | One of the nine intent categories above |

## Usage

```bash
jupyter notebook notebooks/guard.ipynb
```

The notebook uses the `src/` modules to load the data, generate embeddings, build the signed graph, train the model, and produce a `classification_report`, confusion matrix, and false-negative analysis on the test set.

The `src/` modules can also be used independently, for example:

```python
from src.config import Config
from src import data

cfg = Config()
df = data.load_dataset(cfg.DATA_PATH)
```

## Configuration

Key hyperparameters are defined in the `Config` dataclass in `src/config.py`, including graph neighborhood size (`K_GRAPH`), positive and negative similarity thresholds, loss weights (`LAMBDA_*`), propagation coefficients (`PROP_ALPHA_*`), and training parameters (`LR`, `EPOCHS`, and others).

## License

This project is licensed under the [MIT License](LICENSE).

## Disclaimer

This project is a research prototype and is not intended for direct use in real-world medical decision-support systems.
