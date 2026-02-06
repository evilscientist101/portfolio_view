"""
Estimation refinement prototype (simplified): 10 locations, guardrailed adjustments,
portfolio summary + bar chart + scatterplot in sticky top; location table below.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from synthetic_data import get_locations, COL_BASE_KWH, COL_TIER, COL_STEP
from scoring import weighted_net_sd, portfolio_score, score_label

CONCEPT_TEXT = (
    "**How it works:** The **tier** is your guardrail — it defines the range of allowed adjustments "
    "per location. The portfolio-level prediction gives the overall direction; guardrailed adjustments "
    "let you add location-level knowledge the model doesn't see, while staying within the tier and "
    "leveraging the portfolio recommendation. The score encourages balance: don't deviate from the "
    "portfolio-level recommendation on average, but you can refine individual locations within the guardrails."
)

st.set_page_config(page_title="Estimation refinement", layout="wide")

@st.cache_data
def load_ten_locations():
    """Load 10 synthetic locations (no external data)."""
    return get_locations(seed=42)

try:
    df_ten = load_ten_locations()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

n = 10
base_kwh = df_ten[COL_BASE_KWH].values
tier_step = df_ten[COL_STEP].values
display_names = df_ten["Display name"].values

if "refinements" not in st.session_state:
    st.session_state["refinements"] = [0] * n
refinements = st.session_state["refinements"]

# Sticky header: keep title + guardrailed text + metrics + charts visible when scrolling
st.markdown(
    """
    <style>
    /* Stick the header block: next sibling of marker, or next sibling of marker's parent (Streamlit wraps markdown) */
    [data-sticky-portfolio-marker] + *,
    div:has([data-sticky-portfolio-marker]) + * {
        position: sticky !important;
        top: 0 !important;
        z-index: 999 !important;
        background: var(--background-color, #ffffff) !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        border-bottom: 1px solid rgba(128,128,128,0.25) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("Reset all to 0"):
    st.session_state["refinements"] = [0] * n
    st.rerun()

@st.fragment
def locations_table_and_export():
    """Fragment so slider changes only re-run this block (faster)."""
    refinements = st.session_state["refinements"]
    adjusted_kwh = base_kwh + np.array([refinements[i] * tier_step[i] for i in range(n)])
    total_base_f = float(base_kwh.sum())
    total_adjusted_f = float(adjusted_kwh.sum())
    w_net_f = weighted_net_sd(refinements, base_kwh)
    score_f = portfolio_score(w_net_f)
    label_f = score_label(w_net_f)

    # Marker so CSS can target the next element (header container) for sticky
    st.markdown(
        '<div data-sticky-portfolio-marker style="display:none; height:0; margin:0; padding:0;"></div>',
        unsafe_allow_html=True,
    )
    with st.container():
        st.title("Portfolio estimation refinement (4×150 kW)")
        # Guardrailed adjustments at top, always visible (not hidden)
        st.caption("What are guardrailed adjustments?")
        st.markdown(CONCEPT_TEXT)
        # Metrics + charts row
        c1, c2, c3, c4, c_chart1, c_chart2 = st.columns([1, 1, 1, 1, 1.1, 1.1])
        with c1:
            st.metric("Total base kWh/day", f"{total_base_f:,.0f}")
        with c2:
            st.metric("Total adjusted kWh/day", f"{total_adjusted_f:,.0f}")
        with c3:
            st.metric("Portfolio score", f"{score_f:.1f}%")
        with c4:
            st.caption("Net stance")
            st.write(f"**{label_f}**")
        with c_chart1:
            fig_bar, ax_bar = plt.subplots(figsize=(1.3, 0.95))
            ax_bar.bar([0, 1], [total_base_f, total_adjusted_f], width=0.4, color=["steelblue", "seagreen"])
            ax_bar.set_xticks([0, 1])
            ax_bar.set_xticklabels(["Base", "Adj"], fontsize=6)
            ax_bar.tick_params(axis="y", labelsize=5)
            ax_bar.set_ylabel("kWh", fontsize=6)
            plt.tight_layout(pad=0.2)
            st.pyplot(fig_bar)
            plt.close(fig_bar)
        with c_chart2:
            order_r = np.argsort(base_kwh)
            rank = np.empty(n, dtype=int)
            rank[order_r] = np.arange(1, n + 1)
            fig_scatter, ax_scatter = plt.subplots(figsize=(1.3, 0.95))
            for ii in range(n):
                r, y_orig, y_adj = rank[ii], base_kwh[ii], adjusted_kwh[ii]
                ax_scatter.scatter(r, y_orig, c="steelblue", s=8, zorder=2)
                if refinements[ii] != 0:
                    ax_scatter.scatter(r, y_adj, c="seagreen", s=8, zorder=2, marker="s")
                    ax_scatter.plot([r, r], [y_orig, y_adj], "k-", alpha=0.5, lw=0.8, zorder=1)
            ax_scatter.set_xlabel("Rank", fontsize=6)
            ax_scatter.tick_params(labelsize=5)
            plt.tight_layout(pad=0.2)
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)

    # Sort options
    sort_by = st.selectbox("Sort table by", ["Location name", "Tier", "Base kWh/day"], key="sort_by")
    ascending = st.checkbox("Ascending", value=True, key="sort_asc")
    if sort_by == "Location name":
        order = np.argsort(display_names) if ascending else np.argsort(display_names)[::-1]
    elif sort_by == "Tier":
        tiers = df_ten[COL_TIER].values
        order = np.argsort(tiers) if ascending else np.argsort(tiers)[::-1]
    else:
        order = np.argsort(base_kwh) if ascending else np.argsort(base_kwh)[::-1]

    st.subheader("Locations")
    header_cols = st.columns([2.5, 1, 0.8, 0.9, 1.2, 1])
    for h, lab in enumerate(["**Location name**", "**Tier**", "**Base kWh/day**", "**Step (±1) kWh**", "**Estimate adjustment**", "**Adjusted kWh/day**"]):
        with header_cols[h]:
            st.caption(lab)
    for idx in order:
        i = int(idx)
        cols = st.columns([2.5, 1, 0.8, 0.9, 1.2, 1])
        with cols[0]:
            st.write(display_names[i])
        with cols[1]:
            st.write(df_ten[COL_TIER].iloc[i])
        with cols[2]:
            st.write(f"{base_kwh[i]:,.0f}")
        with cols[3]:
            st.write(f"{tier_step[i]:.1f}")
        with cols[4]:
            new_val = st.slider("adj", -3, 3, refinements[i], 1, key=f"adj_{i}", label_visibility="collapsed")
            if new_val != refinements[i]:
                st.session_state["refinements"][i] = new_val
                st.rerun()
        with cols[5]:
            st.write(f"{adjusted_kwh[i]:,.0f}")

    # Build export in same order as table (sorted)
    display_df = pd.DataFrame({
        "Location name": display_names[order],
        "Tier": df_ten[COL_TIER].values[order],
        "Base kWh/day": np.round(base_kwh[order], 1),
        "Step (±1) kWh": np.round(tier_step[order], 1),
        "Estimate adjustment": np.array(refinements)[order],
        "Adjusted kWh/day": np.round(adjusted_kwh[order], 1),
    })
    st.subheader("Export")
    if st.button("Export adjusted estimates to CSV"):
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "adjusted_estimates_10.csv"
        display_df.to_csv(path, index=False)
        st.success(f"Saved to `{path}`.")

locations_table_and_export()
