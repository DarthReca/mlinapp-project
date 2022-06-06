import argparse
import pandas as pd
import numpy as np


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="./data/list_attr_celeba.csv",
        help="the path of the csv containing the annotations)",
    )
    parser.add_argument(
        "-a",
        "--attr",
        type=str,
        required=True,
        help="the attribute whose images' indices will be extracted",
    )
    return parser.parse_args()


def index_from_filename(filename: str):
    # remove ext and map to int (-1 because filenames are 1-indexed)
    without_ext = filename[:-4]
    as_int = int(without_ext)
    return as_int-1


def _main():
    # parsing arguments
    args = _get_args()
    print(f"Filtering '{args.data}' on the '{args.attr}' attribute")

    df = pd.read_csv(args.data)

    try:
        # filter by (specified-attribute == 1)
        df = df.loc[df[args.attr] == 1]
        print(f"Selected {len(df)} images with the '{args.attr}' attribute")

        # retrieve filenames
        filenames = df["Image_Name"].to_list()

        # map each filename to an index
        indices = list(map(index_from_filename, filenames))
        print(f"The first 5 indices are: {indices[:5]}")

        # convert to numpy array and save it
        np_indices = np.asarray(indices)
        np.save(f"indices_{args.attr}", np_indices)
        print(f"Indices saved to 'indices_{args.attr}.npy'")
    except KeyError:
        print(f"The '{args.attr}' attribute isn't valid")


if __name__ == "__main__":
    _main()
