"""
cli.py: command line interface for this package
-----------------------------------------------


* Copyright: 2023 Dat Tran
* Authors: Dat Tran (hello@dats.bio)
* Date: 2023-12-06
* Version: 0.0.1


This is part of the construct_image_blob package


License
-------
Proprietary License

"""

from __future__ import annotations
import os
import argparse
import numpy as np
import joblib
from cvinfer.common import Frame, BinaryBlob
from tqdm import tqdm


def parse_args():
    # Create the main parser
    parser = argparse.ArgumentParser(description="construct-image-blob cli")
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="path to image directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="path to directory",
    )
    parser.add_argument(
        "--nb-blob",
        type=int,
        required=True,
        help="nb of blobs",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="if turned on, data will be saved as compressed",
    )

    # Parse the arguments
    args = parser.parse_args()
    return args


def construct_blob(
    file_per_blob: int, worker_index: int, image_dir: str, output_dir: str, compressed: bool
):
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir) if not f.startswith(".")
    ]
    image_files.sort()
    start_idx = worker_index * file_per_blob
    stop_idx = min((worker_index + 1) * file_per_blob, len(image_files))
    image_files = image_files[start_idx:stop_idx]
    progbar = tqdm(desc=f"Worker {worker_index}", total=len(image_files))
    blob = BinaryBlob(
        binary_file=os.path.join(output_dir, f"{worker_index:09d}.bin"),
        index_file=os.path.join(output_dir, f"{worker_index:09d}.idx"),
        mode="w",
    )
    count = 0
    for img_file in image_files:
        try:
            frame = Frame(img_file)
            assert frame.height() > 0 and frame.width() > 0
        except BaseException:
            continue

        if compressed:
            # read the bytes from file
            with open(img_file, "rb") as f:
                data = f.read()
        else:
            data = Frame(img_file).data()
        blob.write_index(count, data)
        count += 1
        progbar.update(1)
    blob.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(args.image_dir) if not f.startswith(".")]
    if len(image_files) == 0:
        print("No image files found in the directory")
        return

    nb_blob = min(args.nb_blob, len(image_files))
    file_per_blob = int(np.ceil(len(image_files) / nb_blob))

    joblib.Parallel(n_jobs=nb_blob, backend="loky")(
        joblib.delayed(construct_blob)(
            file_per_blob,
            i,
            args.image_dir,
            args.output_dir,
            args.compress,
        )
        for i in range(nb_blob)
    )
    print("Done!")


if __name__ == "__main__":
    main()
