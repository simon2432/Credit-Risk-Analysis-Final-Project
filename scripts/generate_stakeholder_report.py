import json
from datetime import datetime
from pathlib import Path

from src.config import MODEL_DIR


def _find_latest_prediction_stats(model_dir: Path) -> Path:
    files = sorted(model_dir.glob("prediction_stats_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No prediction_stats_*.json found in model directory")
    return files[0]


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main():
    stats_path = _find_latest_prediction_stats(MODEL_DIR)
    payload = json.loads(stats_path.read_text(encoding="utf-8"))

    models = payload.get("models", [])
    test_metrics = payload.get("test_metrics", {})

    report_lines = []
    report_lines.append("Credit Risk Model - Stakeholder Report")
    report_lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Source file: {stats_path}")
    report_lines.append("")
    report_lines.append("Model Comparison (Validation)")
    report_lines.append("------------------------------------------------------------")
    for m in models:
        name = m.get("model_name", "unknown")
        cm = m.get("curve_metrics", {})
        ps = m.get("prediction_stats", {}).get("val", {}).get("threshold_opt", {})
        report_lines.append(f"Model: {name}")
        report_lines.append(f"  ROC AUC: {cm.get('roc_auc', 'n/a')}")
        report_lines.append(f"  Accuracy Ratio: {cm.get('accuracy_ratio', 'n/a')}")
        report_lines.append(f"  Pietra Index: {cm.get('pietra_index', 'n/a')}")
        if ps:
            report_lines.append(f"  Approval Rate (opt threshold): {_fmt_pct(1 - ps.get('pred_rate', 0.0))}")
            report_lines.append(f"  Reject Rate (opt threshold): {_fmt_pct(ps.get('pred_rate', 0.0))}")
        report_lines.append("")

    report_lines.append("Out-of-Sample (Test) Summary - Best Model")
    report_lines.append("------------------------------------------------------------")
    test_curve = test_metrics.get("curve_metrics", {})
    test_pred = test_metrics.get("prediction_stats", {}).get("threshold_opt", {})
    report_lines.append(f"  ROC AUC: {test_curve.get('roc_auc', 'n/a')}")
    report_lines.append(f"  Accuracy Ratio: {test_curve.get('accuracy_ratio', 'n/a')}")
    report_lines.append(f"  Pietra Index: {test_curve.get('pietra_index', 'n/a')}")
    if test_pred:
        report_lines.append(f"  Approval Rate (opt threshold): {_fmt_pct(1 - test_pred.get('pred_rate', 0.0))}")
        report_lines.append(f"  Reject Rate (opt threshold): {_fmt_pct(test_pred.get('pred_rate', 0.0))}")
    report_lines.append("")

    report_lines.append("Notes")
    report_lines.append("------------------------------------------------------------")
    report_lines.append("- Validation metrics reflect model comparison across candidates.")
    report_lines.append("- Test metrics are informational and do not affect model selection.")

    out_path = MODEL_DIR / "stakeholder_report.txt"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[OK] Report saved to: {out_path}")


if __name__ == "__main__":
    main()
