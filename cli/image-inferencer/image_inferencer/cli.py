"""
cli.py: command line interface for this package
-----------------------------------------------


* Copyright: 2023 Dat Tran
* Authors: Dat Tran (hello@dats.bio)
* Date: 2023-12-06
* Version: 0.0.1


This is part of the image_inferencer package


License
-------
Proprietary License

"""

from __future__ import annotations
import argparse
import os
from image_inferencer.blob import process_image_blob
from image_inferencer.image import process_image_dir


def parse_args():
    # Create the main parser
    parser = argparse.ArgumentParser(description="image_inferencer cli")
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["image", "blob"],
        required=True,
        help="type of data: can be either image or blob",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="output directory to save results",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="path to model directory",
    )
    parser.add_argument(
        "--relative-onnx-path",
        default="model.onnx",
        type=str,
        help="path onnx file under asset dir",
    )
    parser.add_argument(
        "--relative-config-path",
        default="configuration.json",
        type=str,
        help="path config file under asset dir",
    )
    parser.add_argument(
        "--relative-processor-path",
        default="processor.py",
        type=str,
        help="path processor file under asset dir",
    )
    parser.add_argument(
        "--mem-limit", default=11, type=int, help="size in GB of gpu memory if using GPU"
    )
    parser.add_argument("--batch-size", default=32, type=int, help="batch size")
    parser.add_argument(
        "--nb-worker",
        default=None,
        type=int,
        help="number of worker. If using GPUs, default to number of GPUs. If using CPUs, default to number of cores",
    )
    parser.add_argument(
        "--execution-provider",
        default="CPUExecutionProvider",
        type=str,
        help="execution provider for onnx",
    )

    # Parse the arguments
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.data_type == "blob":
        process_image_blob(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            asset_dir=args.model_dir,
            relative_onnx_path=args.relative_onnx_path,
            relative_config_path=args.relative_config_path,
            relative_processor_path=args.relative_processor_path,
            mem_limit=args.mem_limit,
            batch_size=args.batch_size,
            nb_worker=args.nb_worker,
            execution_provider=args.execution_provider,
        )
    else:
        process_image_dir(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            asset_dir=args.model_dir,
            relative_onnx_path=args.relative_onnx_path,
            relative_config_path=args.relative_config_path,
            relative_processor_path=args.relative_processor_path,
            mem_limit=args.mem_limit,
            batch_size=args.batch_size,
            nb_worker=args.nb_worker,
            execution_provider=args.execution_provider,
        )


if __name__ == "__main__":
    main()
