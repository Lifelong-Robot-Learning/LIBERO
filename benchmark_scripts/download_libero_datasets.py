import init_path
import argparse
import os

import libero.libero.utils.download_utils as download_utils
from libero.libero import get_libero_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        type=str,
        default=get_libero_path("datasets"),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=["all", "libero_goal", "libero_spatial", "libero_object", "libero_100"],
        default="all",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    # Ask users to specify the download directory of datasets
    os.makedirs(args.download_dir, exist_ok=True)
    print(f"Datasets downloaded to {args.download_dir}")
    print(f"Downloading {args.datasets} datasets")

    # If not, download
    download_utils.libero_dataset_download(
        download_dir=args.download_dir, datasets=args.datasets
    )

    # (TODO) If datasets exist, check if datasets are the same as benchmark

    # Check if datasets exist first
    download_utils.check_libero_dataset(download_dir=args.download_dir)


if __name__ == "__main__":
    main()
