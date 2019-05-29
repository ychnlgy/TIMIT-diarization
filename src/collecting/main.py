import argparse

import save

def main(args):
    if args.load:
        save.load(args.file)
    else:
        save.save(args.root, args.file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--load", type=int, help="0: saves to root, 1: loads from root.")
    parser.add_argument("--root", type=str, help="Directory in which TIMIT is stored.", default="")
    parser.add_argument("--file", type=str, help="File path at which the extracted data is stored.")

    args = parser.parse_args()

    main(args)
