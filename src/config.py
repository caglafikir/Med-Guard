from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    DATA_PATH: str = "../data/dataset_shuffled.csv"
    RANDOM_STATE: int = 123
    EMBED_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"

    TEST_SIZE: float = 0.20
    CAL_SIZE_WITHIN_TRAIN: float = 0.15

    K_GRAPH: int = 15
    POS_SIM_THRESHOLD: float = 0.75
    NEG_SIM_THRESHOLD: float = 0.70
    PUSH_MARGIN: float = 0.60

    LR: float = 0.003
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 200

    LAMBDA_INTENT: float = 0.5
    LAMBDA_PULL: float = 1.0
    LAMBDA_PUSH: float = 0.6
    LAMBDA_PROP: float = 0.5

    POS_WEIGHT: float = 2.0

    PROP_ALPHA_POS: float = 0.3
    PROP_ALPHA_NEG: float = 0.2
    PROP_K: int = 2
    NEG_LAMBDA: float = 0.8

    K_INDUCTIVE: int = 10
    INDUCTIVE_SIM_THR: float = 0.60

    MIN_COVERAGE: float = 0.80
    LOW_GRID: np.ndarray = field(default_factory=lambda: np.arange(0.05, 0.46, 0.02))
    HIGH_GRID: np.ndarray = field(default_factory=lambda: np.arange(0.55, 0.96, 0.02))


LABEL_MAP = {"safe": 0, "unsafe": 1}

INTENT_MAP = {
    "benign_general":        0,
    "medication_related":    1,
    "mild_emotional_stress": 2,
    "chronic_condition":     3,
    "non_acute_injury":      4,
    "acute_medical_risk":    5,
    "harmful_treatment":     6,
    "manipulation":          7,
    "self_harm_risk":        8,
}
IDX_TO_INTENT = {v: k for k, v in INTENT_MAP.items()}

INTENT_CLASS_WEIGHTS = [1.0, 1.5, 1.5, 1.5, 1.5, 5.0, 5.0, 5.0, 10.0]

CONFUSION_PAIR_NAMES = [
    ("acute_medical_risk", "non_acute_injury"),
    ("acute_medical_risk", "chronic_condition"),
    ("harmful_treatment", "benign_general"),
    ("harmful_treatment", "chronic_condition"),
    ("self_harm_risk", "medication_related"),
    ("manipulation", "benign_general"),
]
