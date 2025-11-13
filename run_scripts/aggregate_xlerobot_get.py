import argparse
import os
from pathlib import Path
from typing import Iterable

from lerobot.datasets.aggregate import aggregate_datasets


DEFAULT_DATASETS = [
    "yihao-brain-bot/xlerobot-get-water",
    "yihao-brain-bot/xlerobot-get-hershey",
    "yihao-brain-bot/xlerobot-get-kitkat",
    "yihao-brain-bot/xlerobot-get-musketeers",
    "yihao-brain-bot/xlerobot-get-altereco",
    "yihao-brain-bot/xlerobot-close-cabinet-left3",
    "yihao-brain-bot/xlerobot-close-cabinet-right",
    "yihao-brain-bot/xlerobot-open-cabinet-left2",
    "yihao-brain-bot/xlerobot-open-cabinet-right2",
    "yihao-brain-bot/xlerobot-open-cabinet-right3",
    "yihao-brain-bot/xlerobot-pick-lemon-right2",
    # "yihao-brain-bot/xlerobot-data",
    "yihao-brain-bot/xlerobot-data-2",
]


def find_local_datasets(base_dir: Path, include: Iterable[str] | None) -> list[Path]:
    dataset_names = include if include else DEFAULT_DATASETS
    candidates = [base_dir / name.split("/", maxsplit=1)[1] for name in dataset_names]

    def is_lerobot_dataset(path: Path) -> bool:
        return (path / "meta" / "info.json").exists()

    return sorted(path for path in candidates if is_lerobot_dataset(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate local LeRobot datasets without hitting the Hub")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/home/yihao/.cache/huggingface/lerobot/yihao-brain-bot"),
        help="Directory containing the individual datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional list of dataset folder names to include (defaults to every dataset under base-dir)",
    )
    parser.add_argument(
        "--output-name",
        default="xlerobot",
        help="Name of the aggregated dataset folder to create under base-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    roots = find_local_datasets(base_dir, args.datasets)
    if not roots:
        raise FileNotFoundError("No LeRobot datasets were found under the specified base directory.")

    missing = [path for path in roots if not path.exists()]
    if missing:
        formatted = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "The following dataset directories were not found:\n"
            f"  - {formatted}\n"
            "Please verify the paths before running the aggregation."
        )

    repo_ids = [f"local/{path.name}" for path in roots]

    aggr_root = base_dir / args.output_name
    if aggr_root.exists():
        raise FileExistsError(
            f"Destination directory already exists: {aggr_root}\n"
            "Move or delete it before running the aggregation to avoid overwriting data."
        )

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=f"local/{args.output_name}",
        roots=roots,
        aggr_root=aggr_root,
    )

    print(f"Aggregation complete. New dataset stored at: {aggr_root}")


if __name__ == "__main__":
    main()
