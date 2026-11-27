# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""Neural network definitions: the AI teacher and the small student nets."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (
            nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
            if (cin != cout or stride != 1) else nn.Identity()
        )
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.act(h + self.skip(x))


class Teacher(nn.Module):
    """1D ResNet on 12-lead ECG (12 x T). Outputs a logit for 'older'."""

    def __init__(self, channels: int = 64, blocks: int = 4, n_leads: int = 12):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_leads, channels, 15, 2, 7, bias=False), nn.BatchNorm1d(channels), nn.ReLU())
        layers, c = [], channels
        for i in range(blocks):
            cout = channels * (2 ** min(i, 2))
            layers.append(ResBlock1d(c, cout, 2))
            c = cout
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c, 1))

    def forward(self, x):
        return self.head(self.body(self.stem(x))).squeeze(-1)


class Student(nn.Module):
    """Small CNN learner. `age_head` answers the task; `cue_head` (cue group) predicts cue values;
    `aux_head` is used only for the distractor task before the delayed test."""

    def __init__(self, n_cues: int, channels: int = 16, n_leads: int = 12):
        super().__init__()
        c = channels
        self.backbone = nn.Sequential(
            nn.Conv1d(n_leads, c, 11, 2, 5), nn.ReLU(),
            nn.Conv1d(c, 2 * c, 9, 2, 4), nn.ReLU(),
            nn.Conv1d(2 * c, 2 * c, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(2 * c, 4 * c, 5, 2, 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.age_head = nn.Linear(4 * c, 1)
        self.cue_head = nn.Linear(4 * c, max(n_cues, 1))
        self.aux_head = nn.Linear(4 * c, 1)

    def forward(self, x):
        z = self.backbone(x)
        return self.age_head(z).squeeze(-1), self.cue_head(z), self.aux_head(z).squeeze(-1)
