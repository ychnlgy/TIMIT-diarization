import argparse

from __init__ import src

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--file",
        help = "Data path.",
        default = "../data/timit-wav-mfcc.npy"
    )
    
    parser.add_argument(
        "--repeats",
        type = int,
        help = "Number of random slices to withdraw per sample."
    )
    
    parser.add_argument(
        "--slicelen",
        type = int,
        help = "Length of each random slice.",
        default = 64
    )

    parser.add_argument(
        "--batchsize",
        type = int,
        help = "Training batch size. Note that test batch is double this.",
        default = 32
    ),

    parser.add_argument(
        "--l2",
        type = float,
        help = "L2-regularization weight",
    )

    parser.add_argument(
        "--device",
        default = "cuda"
    )
    
    args = parser.parse_args()

    src.training.train.main(
        fpath = args.file,
        repeats = args.repeats,
        slicelen = args.slicelen,
        batchsize = args.batchsize,
        l2reg = args.l2,
        device = args.device
    )
