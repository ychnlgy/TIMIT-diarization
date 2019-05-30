import argparse, matplotlib, scipy.io.wavfile, numpy
matplotlib.use("agg")
from matplotlib import pyplot

from __init__ import src

def normalize(data):
    data = data.astype(float)
    miu = data.mean()
    std = data.std()
    return (data-miu)/std

def visualize(fpath):
    data, test = src.toolkit.save.load(fpath)
    subject_id, subject_data = next(iter(data.items()))
    sample_id, sample_data = next(iter(subject_data.items()))
    wave = sample_data[src.collecting.WAV_DATA]#.astype(numpy.float32, order="C")
    #wave = wave/wave.max()*2-1
    scipy.io.wavfile.write("sample.wav", 16000, wave)
    
    mfcc = sample_data[src.preprocessing.MFCC]

    fig, axes = pyplot.subplots(nrows=2)
    axes[0].plot(wave)
    print("MFCC shape:", mfcc.shape)
    axes[1].imshow(mfcc.T, interpolation="nearest", cmap="hot", aspect="auto")

    pyplot.savefig(
        "sample.png",#"../data/%s-%s.png" % (subject_id, sample_id),
        bbox_inches="tight"
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, help="File path at which the extracted data is stored.")

    args = parser.parse_args()

    visualize(args.file)
