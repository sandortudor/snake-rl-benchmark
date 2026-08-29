from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_metrics(runs_dir: Path) -> list[dict]:
    rows = []
    for path in runs_dir.glob("*/seed_*/metrics.csv"):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["episode"] = int(row["episode"])
                row["score"] = float(row["score"])
                row["moving_avg_score"] = float(row["moving_avg_score"])
                row["elapsed_seconds"] = float(row["elapsed_seconds"])
                row["model_size"] = int(float(row["model_size"]))
                rows.append(row)
    return rows


def aggregate_by_episode(rows: list[dict]) -> dict[str, list[tuple[int, float, float]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["model"]][row["episode"]].append(row["moving_avg_score"])

    series = {}
    for model, by_episode in grouped.items():
        points = []
        for episode in sorted(by_episode):
            values = by_episode[episode]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            points.append((episode, mean, variance ** 0.5))
        series[model] = points
    return series


def write_svg_line_chart(series: dict[str, list[tuple[int, float, float]]], path: Path, title: str):
    width, height = 1000, 600
    margin = 70
    colors = ["#0f766e", "#2563eb", "#c2410c", "#7c3aed", "#be123c", "#4d7c0f", "#111827"]
    max_x = max(point[0] for points in series.values() for point in points)
    max_y = max(point[1] for points in series.values() for point in points) or 1.0

    def sx(x):
        return margin + (width - margin * 2) * x / max_x

    def sy(y):
        return height - margin - (height - margin * 2) * y / max_y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="14" font-family="Arial">Training episode</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="14" font-family="Arial">Moving average score</text>',
    ]
    for i, (model, points) in enumerate(sorted(series.items())):
        color = colors[i % len(colors)]
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{d}"/>')
        parts.append(f'<text x="{width-margin+10}" y="{margin + i*22}" font-size="14" font-family="Arial" fill="{color}">{model}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary_table(runs_dir: Path, reports_dir: Path):
    summary_path = runs_dir / "benchmark_summary.csv"
    if not summary_path.exists():
        return []
    rows = list(csv.DictReader(summary_path.open(newline="")))
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    table_rows = []
    for model, values in by_model.items():
        final_avg = sum(float(v["final_avg_score"]) for v in values) / len(values)
        best = max(float(v["best_score"]) for v in values)
        seconds = sum(float(v["total_train_seconds"]) for v in values) / len(values)
        size = max(int(float(v["model_size"])) for v in values)
        efficiency = final_avg / max(seconds, 0.001)
        stability = _stddev([float(v["final_avg_score"]) for v in values])
        table_rows.append((model, final_avg, best, seconds, size, efficiency, stability))
    table_rows.sort(key=lambda row: row[5], reverse=True)

    out = reports_dir / "comparison_table.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "model", "avg_final_score", "best_score", "avg_train_seconds", "model_size", "score_per_second", "score_stddev"])
        for rank, row in enumerate(table_rows, start=1):
            writer.writerow([rank, row[0], round(row[1], 4), round(row[2], 4), round(row[3], 4), row[4], round(row[5], 6), round(row[6], 4)])
    return table_rows


def _stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def write_svg_bar_chart(rows: list[tuple], value_index: int, path: Path, title: str, y_label: str):
    width, height = 1000, 600
    margin = 80
    colors = ["#0f766e", "#2563eb", "#c2410c", "#7c3aed", "#be123c", "#4d7c0f", "#111827"]
    values = [float(row[value_index]) for row in rows]
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1.0)
    bar_area = width - margin * 2
    bar_width = bar_area / max(len(rows), 1) * 0.65

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="14" font-family="Arial">{y_label}</text>',
    ]
    for i, row in enumerate(rows):
        model = row[0]
        value = float(row[value_index])
        x = margin + i * (bar_area / max(len(rows), 1)) + bar_width * 0.25
        bar_height = (height - margin * 2) * value / max_value
        y = height - margin - bar_height
        color = colors[i % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="13" font-family="Arial">{value:.2f}</text>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{height - margin + 20}" text-anchor="end" transform="rotate(-35 {x + bar_width/2:.1f} {height - margin + 20})" font-size="12" font-family="Arial">{model}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate report-ready charts from Snake RL runs.")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.reports.mkdir(parents=True, exist_ok=True)
    rows = read_metrics(args.runs)
    if not rows:
        raise SystemExit("No metrics found. Run training first.")
    series = aggregate_by_episode(rows)
    write_svg_line_chart(series, args.reports / "score_over_training.svg", "Snake RL score over training")
    table_rows = write_summary_table(args.runs, args.reports)
    if table_rows:
        write_svg_bar_chart(table_rows, 1, args.reports / "final_score.svg", "Final score comparison", "Average final score")
        write_svg_bar_chart(table_rows, 3, args.reports / "training_time.svg", "Training time comparison", "Average seconds")
        write_svg_bar_chart(table_rows, 4, args.reports / "model_size.svg", "Model size comparison", "Parameters or table entries")
        write_svg_bar_chart(table_rows, 5, args.reports / "score_per_second.svg", "Speed-quality efficiency", "Final score per second")
        write_svg_bar_chart(table_rows, 6, args.reports / "stability.svg", "Training stability", "Final score standard deviation")
    print(f"wrote reports to {args.reports}")


if __name__ == "__main__":
    main()
