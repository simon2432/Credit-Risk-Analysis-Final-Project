"""
Utilities for identifying top-N best payers (lowest default probability)
and estimating margin based on loan amount and interest rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SegmentOutput:
    top_csv_path: str
    margin_csv_path: str
    scatter_png_path: str


def _as_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def build_top_payers(
    X: pd.DataFrame,
    pred_proba: Sequence[float],
    *,
    top_pct: float,
    out_dir: str | Path,
    prefix: str = "top_payers",
    amount: Optional[float] = None,
    rate: Optional[float] = None,
) -> SegmentOutput:
    """
    Build a Top-N% best payers segment based on lowest default probability.

    Inputs:
      - X: DataFrame with loan attributes
      - pred_proba: model P(default=1) for each row
      - amount_col: column name for loan amount
      - rate_col: column name for interest rate (as decimal, e.g., 0.25)
      - top_pct: percentage in [0, 100], e.g., 5 or 10
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = X.copy()
    df["pred_proba"] = _as_array(pred_proba)

    if amount is None or rate is None:
        raise ValueError("amount and rate must be provided for scenario calculations")
    df["amount"] = float(amount)
    df["rate"] = float(rate)

    # Expected margin proxy = amount * rate * (1 - default_prob)
    df["expected_margin"] = df["amount"] * df["rate"] * (1.0 - df["pred_proba"])

    # Lower default probability = better payer
    df = df.sort_values("pred_proba", ascending=True).reset_index(drop=True)
    n_top = max(int(len(df) * (top_pct / 100.0)), 1)
    top_df = df.head(n_top)

    # Save Top-N%
    top_csv = out_dir / f"{prefix}_{int(top_pct)}pct.csv"
    top_df.to_csv(top_csv, index=False)

    # Margin summary
    summary = pd.DataFrame(
        [
            {
                "top_pct": float(top_pct),
                "rows": int(len(top_df)),
                "avg_pd": float(top_df["pred_proba"].mean()),
                "avg_amount": float(top_df["amount"].mean()),
                "avg_rate": float(top_df["rate"].mean()),
                "total_expected_margin": float(top_df["expected_margin"].sum()),
            }
        ]
    )
    margin_csv = out_dir / f"{prefix}_{int(top_pct)}pct_margin.csv"
    summary.to_csv(margin_csv, index=False)

    # Scatter plot: PD vs expected margin
    plt.figure(figsize=(7, 4))
    plt.scatter(df["pred_proba"], df["expected_margin"], s=8, alpha=0.35, label="All")
    plt.scatter(
        top_df["pred_proba"],
        top_df["expected_margin"],
        s=12,
        alpha=0.7,
        label=f"Top {int(top_pct)}%",
        color="#f58518",
    )
    plt.xlabel("Predicted default probability")
    plt.ylabel("Expected margin (amount * rate * (1 - PD))")
    plt.title(f"Top {int(top_pct)}% Best Payers")
    plt.legend()
    plt.tight_layout()
    scatter_png = out_dir / f"{prefix}_{int(top_pct)}pct_scatter.png"
    plt.savefig(scatter_png)
    plt.close()

    return SegmentOutput(
        top_csv_path=str(top_csv),
        margin_csv_path=str(margin_csv),
        scatter_png_path=str(scatter_png),
    )


def build_multiple_top_payers(
    X: pd.DataFrame,
    pred_proba: Sequence[float],
    *,
    top_pcts: Iterable[float],
    out_dir: str | Path,
    prefix: str = "top_payers",
    amount: Optional[float] = None,
    rate: Optional[float] = None,
) -> list[SegmentOutput]:
    results = []
    for pct in top_pcts:
        results.append(
            build_top_payers(
                X,
                pred_proba,
                top_pct=pct,
                out_dir=out_dir,
                prefix=prefix,
                amount=amount,
                rate=rate,
            )
        )
    return results


def build_top_payers_scenarios(
    X: pd.DataFrame,
    pred_proba: Sequence[float],
    *,
    top_pcts: Iterable[float],
    amounts: Iterable[float],
    rates: Iterable[float],
    out_dir: str | Path,
    prefix: str = "top_payers",
) -> pd.DataFrame:
    """
    Scenario grid for multiple top-N%, amounts and rates.
    Writes:
      - top_payers_<pct>pct.csv per segment (using first scenario only)
      - top_payers_<pct>pct_scatter.png per segment (using first scenario only)
      - scenarios_summary.csv (all combinations)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = X.copy()
    df["pred_proba"] = _as_array(pred_proba)
    df = df.sort_values("pred_proba", ascending=True).reset_index(drop=True)

    results = []
    amounts_list = list(amounts)
    rates_list = list(rates)

    for pct in top_pcts:
        n_top = max(int(len(df) * (pct / 100.0)), 1)
        top_df = df.head(n_top).copy()
        rest_df = df.iloc[n_top:].copy()

        # Save Top-N% base file once (no amount/rate dependence)
        top_csv = out_dir / f"{prefix}_{int(pct)}pct.csv"
        top_df.to_csv(top_csv, index=False)

        # Use first scenario for scatterplot
        base_amount = float(amounts_list[0])
        base_rate = float(rates_list[0])
        top_df["amount"] = base_amount
        top_df["rate"] = base_rate
        top_df["expected_margin"] = top_df["amount"] * top_df["rate"] * (1.0 - top_df["pred_proba"])

        plt.figure(figsize=(7, 4))
        plt.scatter(df["pred_proba"], np.zeros_like(df["pred_proba"]), s=6, alpha=0.15, label="All")
        plt.scatter(
            top_df["pred_proba"],
            top_df["expected_margin"],
            s=10,
            alpha=0.7,
            label=f"Top {int(pct)}%",
            color="#f58518",
        )
        plt.xlabel("Predicted default probability")
        plt.ylabel("Expected margin (amount * rate * (1 - PD))")
        plt.title(f"Top {int(pct)}% Best Payers (example scenario)")
        plt.legend()
        plt.tight_layout()
        scatter_png = out_dir / f"{prefix}_{int(pct)}pct_scatter.png"
        plt.savefig(scatter_png)
        plt.close()

        # Additional scatter: PD distribution (Top-N vs Rest)
        plt.figure(figsize=(7, 3.5))
        plt.scatter(rest_df["pred_proba"], np.zeros_like(rest_df["pred_proba"]), s=6, alpha=0.25, label="Rest")
        plt.scatter(top_df["pred_proba"], np.ones_like(top_df["pred_proba"]), s=10, alpha=0.7, label=f"Top {int(pct)}%")
        plt.xlabel("Predicted default probability")
        plt.yticks([0, 1], ["Rest", f"Top {int(pct)}%"])
        plt.title(f"PD Distribution: Top {int(pct)}% vs Rest")
        plt.legend()
        plt.tight_layout()
        pd_scatter_png = out_dir / f"{prefix}_{int(pct)}pct_pd_scatter.png"
        plt.savefig(pd_scatter_png)
        plt.close()

        for amount in amounts_list:
            for rate in rates_list:
                expected_margin = float(amount) * float(rate) * (1.0 - top_df["pred_proba"])
                results.append(
                    {
                        "top_pct": float(pct),
                        "amount": float(amount),
                        "rate": float(rate),
                        "rows": int(len(top_df)),
                        "avg_pd": float(top_df["pred_proba"].mean()),
                        "total_expected_margin": float(expected_margin.sum()),
                        "avg_expected_margin": float(expected_margin.mean()),
                    }
                )

    summary_df = pd.DataFrame(results)
    out_path = out_dir / "scenarios_summary.csv"
    summary_df.to_csv(out_path, index=False)
    return summary_df
