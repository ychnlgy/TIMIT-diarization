import argparse

from __init__ import src

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--file",
        help = "Data path."
    )
    
    parser.add_argument(
        "--repeats",
        type = int,
        help = "Number of random slices to withdraw per sample."
    )
    
    parser.add_argument(
        "--slicelen",
        type = int,
        help = "Length of each random slice."
    )

    parser.add_argument(
        "--batchsize",
        type = int,
        help = "Training batch size. Note that test batch is double this."
    ),

    parser.add_argument(
        "--device",
        default = "cuda"
    )
    
    args = parser.parse_args()

    src.training.train.main(args.file, args.repeats, args.slicelen, args.batchsize, args.device)
