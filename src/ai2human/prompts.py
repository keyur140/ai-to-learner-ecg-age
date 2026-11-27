# AI-to-learner knowledge transfer study (ECG age)
# Author: KEYUR SHAILESHKUMAR PATEL (ORCID: https://orcid.org/0009-0008-5261-043X)
# MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK; Independent researcher
# Conducted independently; not an official Brunel University London project. Licence: MIT
"""All text shown to language-model learners. Fixed in the preregistration."""
from __future__ import annotations

import numpy as np

INTRO = (
    "You are a student learning a new skill. You will see measurements taken from adults' resting 12-lead ECGs. "
    "Each ECG belongs either to a younger adult (under 40 years old) or to an older adult (65 years or older). "
    "Exactly half of all cases are younger and half are older."
)

PANEL_DESCRIPTIONS = {
    "HR_Ventr_Global": "Heart rate is the number of ventricular beats per minute.",
    "PR_Int_Global": "The PR interval is the time from the start of the P wave to the start of the QRS complex.",
    "QRS_Dur_Global": "QRS duration is how long the ventricles take to depolarise.",
    "QT_Int_Global": "The QT interval runs from the start of the QRS complex to the end of the T wave.",
    "QT_IntBazett_Global": "QTc is the QT interval corrected for heart rate with Bazett's formula.",
    "P_AxisFront_Global": "The P-wave axis is the main direction of atrial depolarisation in the frontal plane.",
    "R_AxisFrontal_Global": "The QRS axis is the main direction of ventricular depolarisation in the frontal plane.",
    "T_AxisFront_Global": "The T-wave axis is the main direction of ventricular repolarisation in the frontal plane.",
    "P_Amp_II": "P-wave height in lead II is the peak voltage of the atrial wave.",
    "P_Dur_II": "P-wave duration in lead II is how long the atrial wave lasts.",
    "R_Amp_aVL": "R-wave height in aVL is the peak upward QRS voltage in that lead.",
    "R_Amp_V5": "R-wave height in V5 is the peak upward QRS voltage in that chest lead.",
    "S_Amp_V1": "S-wave depth in V1 is the downward QRS voltage in that chest lead.",
    "T_Amp_II": "T-wave height in lead II is the peak voltage of the repolarisation wave.",
    "T_Amp_V1": "T-wave height in V1 is the peak repolarisation voltage in that chest lead.",
    "T_Amp_V5": "T-wave height in V5 is the peak repolarisation voltage in that chest lead.",
    "ST_Amp_V5": "The ST level in V5 is the voltage at the J point compared with baseline.",
    "Q_Amp_III": "Q-wave depth in lead III is the first downward QRS voltage in that lead.",
}


def neutral_lesson(panel: list[dict], target_words: int) -> str:
    """Placebo lesson: describes what the measurements are, with no age information.
    Length-matched (in words) to the cue lesson."""
    lines = ["LESSON: What the ECG measurements mean."]
    i = 0
    while sum(len(l.split()) for l in lines) < target_words:
        p = panel[i % len(panel)]
        lines.append(f"- {PANEL_DESCRIPTIONS.get(p['id'], p['label'] + ' is one of the standard ECG measurements.')}")
        i += 1
        if i > 4 * len(panel):
            break
    return "\n".join(lines)


def fmt_value(v: float, unit: str) -> str:
    return f"{v:.2f} {unit}" if unit == "mV" else f"{v:.0f} {unit}"


def format_case(row: np.ndarray, panel: list[dict], blind: dict | None = None) -> str:
    """One ECG as text. With `blind`, names become 'Measure k' and units are dropped."""
    if blind:
        return "\n".join(f"{blind[p['id']]}: {fmt_value(float(v), p['unit']).split(' ')[0]}" for p, v in zip(panel, row))
    return "\n".join(f"{p['label']}: {fmt_value(float(v), p['unit'])}" for p, v in zip(panel, row))


def neutral_lesson_blinded(n_measures: int, target_words: int) -> str:
    """Placebo lesson for the blinded run: describes the layout only, no meaning, no age information."""
    lines = ["LESSON: How the cases are shown.",
             f"- Each case lists {n_measures} measurements, labelled Measure 1 to Measure {n_measures}.",
             "- The measurements are always listed in the same order.",
             "- Some measurements are times, some are angles and some are voltages; units are not shown.",
             "- Values differ from person to person, and no single value is shown on its own as decisive.",
             "- Read every measurement in a case before deciding."]
    k = 1
    while sum(len(l.split()) for l in lines) < target_words and k <= n_measures:
        lines.append(f"- Measure {k} is reported for every case.")
        k += 1
    return "\n".join(lines)


def gate_intro(rule_label: str, threshold_txt: str) -> str:
    """Known-answer gate: the answer is defined by one explicit rule stated in the prompt."""
    return (INTRO + "\n\nIn this exercise the answer is decided by ONE rule only: a case is an older adult "
            f"(65 or over) exactly when {rule_label} is above {threshold_txt}; otherwise it is a younger adult (under 40).")


def probe_question(label: str, higher_is_a: bool = True) -> str:
    """Prior-knowledge probe: what does the model already believe about one measurement?
    Asked twice (v0.5.1, decision D18; issue P20): once with A = higher, once with A = lower."""
    first, second = ("higher", "lower") if higher_is_a else ("lower", "higher")
    return (f"Question about adults' resting ECGs: is the {label} usually higher or lower in adults aged 65 or over "
            f"than in adults under 40?\nA) {first} in older adults\nB) {second} in older adults\nAnswer with the single letter A or B.")


def probe_known(p_higher_a_first: float, p_higher_b_first: float) -> int:
    """Direction a model 'knows' from the two orderings: +1 higher, -1 lower, 0 not consistent (the
    two orderings disagree, as a model that always picks one letter would)."""
    if p_higher_a_first > 0.5 and p_higher_b_first > 0.5:
        return 1
    if p_higher_a_first < 0.5 and p_higher_b_first < 0.5:
        return -1
    return 0


FORBIDDEN = ["ecg_id", "patient", "strat_fold", "device", "site", "recording date", "male", "female", "sex:", "nurse"]


def leak_scan_case(case_txt: str, allowed_labels: set[str]) -> list[str]:
    """A case may contain only 'label: number [unit]' lines with preregistered labels."""
    import re

    problems = []
    for line in case_txt.splitlines():
        m = re.fullmatch(r"(.+?): (-?\d+(?:\.\d+)?)(?: (bpm|ms|deg|mV))?", line)
        if not m or m.group(1) not in allowed_labels:
            problems.append(line)
    return problems


def leak_scan_context(text: str) -> list[str]:
    import re

    low = text.lower()
    return [w for w in FORBIDDEN if re.search(r"\b" + re.escape(w.rstrip(":")) + r"\b", low)]


class Letters:
    """Per-learner mapping of answer letters to classes (counterbalanced)."""

    NAMES = {0: "younger adult (under 40)", 1: "older adult (65 or over)"}

    def __init__(self, older_is_a: bool):
        self.letter = {1: "A", 0: "B"} if older_is_a else {0: "A", 1: "B"}

    def options(self) -> str:
        inv = {v: k for k, v in self.letter.items()}
        return f"A) {self.NAMES[inv['A']]}\nB) {self.NAMES[inv['B']]}"

    def answer(self, label: int) -> str:
        return f"{self.letter[label]}) {self.NAMES[label]}"


def practice_block(cases: list[tuple[str, int]], letters: Letters, title: str) -> str:
    out = [title]
    for k, (txt, lab) in enumerate(cases, 1):
        out.append(f"\nCase {k}:\n{txt}\nCorrect answer: {letters.answer(lab)}")
    return "\n".join(out)


def distractor_block(rng: np.random.Generator, target_tokens: int) -> str:
    """Interference task: weather-station readings with feedback (unrelated to ECGs)."""
    out = ["A DIFFERENT TASK: decide whether it rained at a weather station that day. Feedback is given."]
    n = max(1, int(target_tokens / 45))
    for k in range(1, n + 1):
        hum, press, cloud = rng.integers(20, 100), rng.integers(980, 1040), rng.integers(0, 9)
        rain = (hum > 70 and press < 1010) or cloud >= 7
        out.append(f"Day {k}: humidity {hum}%, pressure {press} hPa, cloud cover {cloud}/8. "
                   f"Answer: {'rain' if rain else 'no rain'}.")
    out.append("End of the different task. Now return to the ECG task.")
    return "\n".join(out)


def test_question(case_txt: str, letters: Letters) -> str:
    return f"TEST CASE:\n{case_txt}\n\nWhich adult does this ECG belong to?\n{letters.options()}\nAnswer with the single letter A or B."
