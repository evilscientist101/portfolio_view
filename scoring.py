"""
Portfolio score: weighted net SD and score formula.
Score 100 when balanced; increases when conservative, decreases when aggressive.
"""
import numpy as np


def weighted_net_sd(refinements, base_kwh):
    """
    Weighted net SD so high-base locations have more impact.
    refinements: array-like of SD adjustments per location (-3..+3)
    base_kwh: array-like of base kWh/day per location
    Returns: scalar (can be negative = conservative, positive = aggressive)
    """
    refinements = np.asarray(refinements, dtype=float)
    base_kwh = np.asarray(base_kwh, dtype=float)
    total = base_kwh.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(refinements * base_kwh) / total)


def portfolio_score(weighted_net_sd_value, k=15.0):
    """
    score = 100 - k * weighted_net_sd
    So +1 SD everywhere -> weighted_net_sd = 1 -> score = 100 - k (e.g. 85 if k=15).
    No refinement -> 100. Conservative (negative net) -> > 100. Aggressive -> < 100.
    """
    return 100.0 - k * weighted_net_sd_value


def score_label(weighted_net_sd_value):
    """Short label: conservative / balanced / aggressive. Wider bands so few adjustments don't flip to aggressive."""
    if weighted_net_sd_value < -0.2:
        return "conservative"
    if weighted_net_sd_value > 0.2:
        return "aggressive"
    return "balanced"
