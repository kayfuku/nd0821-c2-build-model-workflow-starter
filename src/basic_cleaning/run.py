#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import os
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info("Dropping outliers")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review type to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Remember to use index=False when saving to CSV, otherwise the data checks
    # in the next step might fail because there will be an extra index column.
    filename = "preprocessed_data.csv"
    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="name for input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="name for cleaned data artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="type for cleaned data artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="description for  cleaned data artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="minimum price to cut outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="maximum price to cut outliers",
        required=True
    )

    args = parser.parse_args()

    go(args)
