from collections import defaultdict
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = {
    "engagement": {
        "title": "Вовлечённость",
        "labels": ["Low", "Medium", "High"],
        "scores": [0.0, 0.5, 1.0],
        "ylabel": "Оценка вовлечённости",
        "boundaries": [0.25, 0.75],
        "regions": ["Низкая", "Средняя", "Высокая"]
    },
    "activation": {
        "title": "Активация",
        "labels": ["Low", "Medium", "High"],
        "scores": [0.0, 0.5, 1.0],
        "ylabel": "Оценка активации",
        "boundaries": [0.25, 0.75],
        "regions": ["Низкая", "Средняя", "Высокая"]
    },
    "valence": {
        "title": "Валентность",
        "labels": ["Negative", "Neutral", "Positive"],
        "scores": [-1.0, 0.0, 1.0],
        "ylabel": "Оценка валентности",
        "boundaries": [-0.5, 0.5],
        "regions": ["Отрицательная", "Нейтральная", "Положительная"]
    }
}


def calculate_score(probabilities, task):
    """Рассчитывает числовую оценку по вероятностям классов."""

    config = TASKS[task]
    labels = config["labels"]
    scores = config["scores"]

    probs = np.array(
        [probabilities[label] for label in labels],
        dtype=float
    )

    return float(np.dot(probs, scores))


def prepare_analysis_rows(records):
    """Преобразует вероятности в индивидуальные и групповые оценки."""

    individual_rows = []

    for record in records:
        row = {
            "timestamp_sec": record["timestamp_sec"],
            "person_id": record["person_id"]
        }

        for task, config in TASKS.items():
            probabilities = record.get(f"{task}_probs")

            if probabilities is None:
                row[f"{task}_score"] = None
                row[f"{task}_class"] = None

                for label in config["labels"]:
                    row[f"{task}_prob_{label}"] = None

                continue

            labels = config["labels"]

            probs = np.array(
                [probabilities[label] for label in labels],
                dtype=float
            )

            row[f"{task}_score"] = calculate_score(
                probabilities, task
            )

            row[f"{task}_class"] = labels[np.argmax(probs)]

            for label, probability in zip(labels, probs):
                row[f"{task}_prob_{label}"] = float(probability)

        individual_rows.append(row)

    # Группируем записи по временным отметкам
    grouped = defaultdict(list)

    for row in individual_rows:
        grouped[row["timestamp_sec"]].append(row)

    group_rows = []

    for timestamp, rows in grouped.items():
        group_row = {
            "timestamp_sec": timestamp,
            "person_id": "GROUP"
        }

        for task, config in TASKS.items():
            available_scores = [
                row[f"{task}_score"]
                for row in rows
                if row[f"{task}_score"] is not None
            ]

            if not available_scores:
                group_row[f"{task}_score"] = None
                group_row[f"{task}_class"] = None

                for label in config["labels"]:
                    group_row[f"{task}_prob_{label}"] = None

                continue

            # Средние вероятности между участниками
            mean_probs = np.mean(
                [
                    [
                        row[f"{task}_prob_{label}"]
                        for label in config["labels"]
                    ]
                    for row in rows
                    if row[f"{task}_score"] is not None
                ],
                axis=0
            )

            group_row[f"{task}_score"] = float(
                np.dot(mean_probs, config["scores"])
            )

            group_row[f"{task}_class"] = config["labels"][
                np.argmax(mean_probs)
            ]

            for label, probability in zip(
                config["labels"], mean_probs
            ):
                group_row[f"{task}_prob_{label}"] = float(probability)

        group_rows.append(group_row)

    return individual_rows, group_rows


def export_csv(individual_rows, group_rows, csv_path):
    """Сохраняет индивидуальные и групповые оценки в CSV."""

    fields = [
        "timestamp_sec",
        "person_id"
    ]

    for task, config in TASKS.items():
        fields.extend([
            f"{task}_score",
            f"{task}_class"
        ])

        fields.extend([
            f"{task}_prob_{label}"
            for label in config["labels"]
        ])

    all_rows = individual_rows + group_rows

    all_rows.sort(
        key=lambda row: (
            row["timestamp_sec"],
            str(row["person_id"]) != "GROUP",
            str(row["person_id"])
        )
    )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


def export_plot(individual_rows, group_rows, png_path):
    # Однократная группировка индивидуальных оценок
    individual_data = {
        task: defaultdict(list)
        for task in TASKS
    }

    group_data = {
        task: []
        for task in TASKS
    }

    for row in individual_rows:
        for task in TASKS:
            score = row[f"{task}_score"]
            if score is not None:
                individual_data[task][row["person_id"]].append(
                    (row["timestamp_sec"], score)
                )

    for row in group_rows:
        for task in TASKS:
            score = row[f"{task}_score"]
            if score is not None:
                group_data[task].append(
                    (row["timestamp_sec"], score)
                )

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 12),
        sharex=True
    )

    for ax, (task, config) in zip(axes, TASKS.items()):
        min_score = min(config["scores"])
        max_score = max(config["scores"])
        boundaries = config["boundaries"]
        regions = config["regions"]

        # Зоны классов
        region_edges = [
            min_score,
            *boundaries,
            max_score
        ]

        for i, region in enumerate(regions):
            ax.axhspan(
                region_edges[i],
                region_edges[i + 1],
                alpha=0.10,
                color=f"C{i}",
                zorder=0
            )

            ax.text(
                1.01,
                (region_edges[i] + region_edges[i + 1]) / 2,
                region,
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=9
            )

        # Границы зон
        for boundary in boundaries:
            ax.axhline(
                boundary,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.7
            )

        # Индивидуальные линии
        for person_id, points in individual_data[task].items():
            points.sort(key=lambda x: x[0])

            times = [p[0] for p in points]
            scores = [p[1] for p in points]

            ax.plot(
                times,
                scores,
                linewidth=1,
                alpha=0.55,
                label=f"Участник {person_id}"
            )

        # Групповая линия
        points = group_data[task]
        points.sort(key=lambda x: x[0])

        if points:
            times = [p[0] for p in points]
            scores = [p[1] for p in points]

            ax.plot(
                times,
                scores,
                color="black",
                linewidth=2.5,
                label="Среднее по группе"
            )

        ax.set_title(config["title"])
        ax.set_ylabel(config["ylabel"])

        ax.set_ylim(
            min_score - 0.05 * (max_score - min_score),
            max_score + 0.05 * (max_score - min_score)
        )

        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=3)

    axes[-1].set_xlabel("Время видео, сек.")

    fig.suptitle(
        "Динамика вовлечённости и эмоциональных состояний",
        fontsize=15
    )

    fig.tight_layout(rect=[0, 0, 0.88, 0.97])
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_dynamics(records, png_path, csv_path):
    """Полный экспорт анализа групповой динамики."""

    if not records:
        raise ValueError("Нет данных для экспорта")

    individual_rows, group_rows = prepare_analysis_rows(records)

    export_csv(individual_rows, group_rows, csv_path)
    export_plot(individual_rows, group_rows, png_path)