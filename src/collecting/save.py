import load, numpy

def save(datadir, savepath):
    data, test = load.load(datadir)
    with open(savepath, "wb") as f:
        numpy.save(f, [data, test])

def load(savepath):
    with open(savepath, "rb") as f:
        numpy.load(f)
