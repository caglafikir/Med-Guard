import torch.nn as nn


class SignedGraphSafetyModel(nn.Module):
    def __init__(self, in_dim, num_intents=9):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.safety_head = nn.Linear(512, 1)
        self.intent_head  = nn.Linear(512, num_intents)

    def forward(self, x):
        features      = self.shared(x)
        safety_logits = self.safety_head(features).squeeze(-1)
        intent_logits = self.intent_head(features)
        return features, safety_logits, intent_logits

    def forward_from_features(self, features):
        safety_logits = self.safety_head(features).squeeze(-1)
        intent_logits = self.intent_head(features)
        return safety_logits, intent_logits
