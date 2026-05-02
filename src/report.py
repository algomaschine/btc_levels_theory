import os

def write_report(out_dir: str, metrics: dict, tests: dict):
    path = os.path.join(out_dir, "report.md")

    lines = []
    lines.append("# LPM Support/Resistance Report\n")
    lines.append("## Model metrics (walk-forward logistic regression)\n")
    for k, v in (metrics or {}).items():
        if k == "windows":
            continue
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Basic dataset stats\n")
    for k, v in (tests or {}).items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Interpretation guide\n")
    lines.append("- Lower **brier_mean** and **logloss_mean** indicate better calibrated predictions.")
    lines.append("- If out-of-sample performance is indistinguishable from a shuffled-label baseline, the theory is falsified.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
