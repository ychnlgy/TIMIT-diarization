import argparse, matplotlib, scipy.io.wavfile, numpy
matplotlib.use("agg")
from matplotlib import pyplot

from __init__ import src

def visualize(fpath, mfccpath):
    print("Loading audio...")
    data, test = src.toolkit.save.load(fpath)
    subject_id, subject_data = next(iter(data.items()))
    sample_id, sample_data = next(iter(subject_data.items()))
    wave = sample_data[src.collecting.WAV_DATA]

    print("Loading MFCCs...")
    data, test = src.toolkit.save.load(mfccpath)
    mfcc = data[subject_id][sample_id][src.preprocessing.MFCC]

    fig, axes = pyplot.subplots(nrows=2)
    axes[0].plot(wave)
    print("MFCC shape:", mfcc.shape)
    axes[1].imshow(mfcc.T, interpolation="nearest", cmap="hot", aspect="auto")

    scipy.io.wavfile.write("sample.wav", 16000, wave)
    pyplot.savefig(
        "sample.png",
        bbox_inches="tight"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, help="File path at which the extracted data is stored.")
    parser.add_argument("--mfcc", type=str, help="File path at which the MFCCs are stored.")

    args = parser.parse_args()

    visualize(args.file, args.mfcc)
