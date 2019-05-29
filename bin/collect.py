import argparse

from __init__ import src

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, help="Directory in which TIMIT is stored.")
    parser.add_argument("--file", type=str, help="File path at which the extracted data is stored.")

    args = parser.parse_args()

    src.collecting.save(args.root, args.file)
