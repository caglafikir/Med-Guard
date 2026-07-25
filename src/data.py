import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from .config import LABEL_MAP, INTENT_MAP


def load_dataset(data_path):
    df = pd.read_csv(data_path)
    df["label"] = df["safety_level"].map(LABEL_MAP)
    df["intent_idx"] = df["intent"].map(INTENT_MAP)
    return df


def embed_texts(texts, embed_model_name):
    encoder = SentenceTransformer(embed_model_name)
    embeddings = encoder.encode(texts, show_progress_bar=True)
    return np.array(embeddings), encoder


def split_train_cal_test(embeddings, labels, intent_labels, texts,
                          test_size, cal_size_within_train, random_state):
    (X_train_full, X_test,
     y_train_full, y_test,
     int_train_full, int_test,
     texts_train_full, texts_test) = train_test_split(
        embeddings, labels, intent_labels, texts,
        test_size=test_size, random_state=random_state, stratify=intent_labels)

    (X_train, X_cal,
     y_train, y_cal,
     int_train, int_cal,
     texts_train, texts_cal) = train_test_split(
        X_train_full, y_train_full, int_train_full, texts_train_full,
        test_size=cal_size_within_train, random_state=random_state,
        stratify=int_train_full)

    return {
        "X_train": X_train, "X_cal": X_cal, "X_test": X_test,
        "y_train": y_train, "y_cal": y_cal, "y_test": y_test,
        "int_train": int_train, "int_cal": int_cal, "int_test": int_test,
        "texts_train": texts_train, "texts_cal": texts_cal, "texts_test": texts_test,
    }


def to_tensors(split, device):
    t = {}
    t["X_train_t"]   = torch.tensor(split["X_train"], dtype=torch.float32, device=device)
    t["y_train_t"]   = torch.tensor(split["y_train"], dtype=torch.float32, device=device)
    t["int_train_t"] = torch.tensor(split["int_train"], dtype=torch.long, device=device)

    t["X_cal_t"] = torch.tensor(split["X_cal"], dtype=torch.float32, device=device)
    t["y_cal_t"] = torch.tensor(split["y_cal"], dtype=torch.float32, device=device)

    t["X_test_t"]   = torch.tensor(split["X_test"], dtype=torch.float32, device=device)
    t["y_test_t"]   = torch.tensor(split["y_test"], dtype=torch.float32, device=device)
    t["int_test_t"] = torch.tensor(split["int_test"], dtype=torch.long, device=device)
    return t
