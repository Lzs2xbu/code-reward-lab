"""
Compatibility wrapper for the old two-step MBPP v2 preprocessing command.

The canonical entrypoint is now:

  python data/prepare_mbpp.py --output_dir data/mbpp_v2

The old `--input_dir` argument is accepted for command compatibility, but no
intermediate v1 parquet is required anymore.
"""

import argparse

from prepare_mbpp import write_mbpp_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/mbpp", help="Ignored; kept for old commands.")
    parser.add_argument("--output_dir", default="data/mbpp_v2")
    parser.add_argument("--dataset_name", default="google-research-datasets/mbpp")
    parser.add_argument("--dataset_config", default="full")
    parser.add_argument("--preview_rows", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("data/prepare_mbpp_v2.py is a compatibility wrapper.")
    print("Use data/prepare_mbpp.py directly for new runs.")
    write_mbpp_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        legacy_no_function_name=False,
        preview_rows=args.preview_rows,
    )


if __name__ == "__main__":
    main()
