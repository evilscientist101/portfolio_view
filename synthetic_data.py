"""
Synthetic dataset for the estimation refinement prototype.
No external files; all data is generated in code with a fixed seed for reproducibility.
"""
import pandas as pd
import numpy as np

LOCATION_NAMES = [
    "Northgate Mall",
    "Riverside Hub",
    "Central Plaza",
    "Westside Station",
    "Lakeside Drive",
    "Downtown Core",
    "Eastgate Plaza",
    "Southpoint Hub",
    "Metro Central",
    "Harbor View",
]

# Column names used by the app
COL_BASE_KWH = "Base kWh/day"
COL_TIER = "tier"
COL_STEP = "tier_step"


def get_locations(seed: int = 42) -> pd.DataFrame:
    """
    Generate 10 synthetic locations with base kWh/day, tier (A/B/C), and step size.
    Step size = max(10, 20% of base) per location.
    """
    rng = np.random.default_rng(seed)
    # Base kWh: mix of low, mid, high to get variety of tiers (e.g. A >= 350, B >= 150, else C)
    base_kwh = np.round(
        rng.uniform(low=80, high=420, size=10), 1
    )
    tiers = []
    for k in base_kwh:
        if k >= 350:
            tiers.append("A")
        elif k >= 150:
            tiers.append("B")
        else:
            tiers.append("C")
    tier_step = np.maximum(10.0, 0.20 * base_kwh)
    return pd.DataFrame({
        "Display name": LOCATION_NAMES,
        COL_BASE_KWH: base_kwh,
        COL_TIER: tiers,
        COL_STEP: np.round(tier_step, 1),
    })
