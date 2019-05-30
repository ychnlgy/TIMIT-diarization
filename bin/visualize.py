import argparse, matplotlib
matplotlib.use("agg")
from matplotlib import pyplot

from __init__ import src

def visualize(fpath):
    data, test = src.toolkit.save.load(fpath)
    subject_id, subject_data = next(iter(data.items()))
    sample_id, sample_data = next(iter(subject_data.items()))
    wave = sample_data[src.collecting.WAV_DATA]
    mfcc = sample_data[src.preprocessing.MFCC]

    axes, fig = pyplot.subplots(nrows=2)
    axes[0].plot(wave)
    axes[1].imshow(mfcc.T, interpolation="nearest", cmap="hot")

    pyplot.savefig(
        "../data/%s-%s.png" % (subject_id, sample_id),
        bbox_inches="tight"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, help="File path at which the extracted data is stored.")

    args = parser.parse_args()

    visualize(args.file)
