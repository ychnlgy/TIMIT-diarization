import numpy

def save(obj, fpath):
    with open(fpath, "wb") as f:
        numpy.save(f, obj)

def load(fpath):
    with open(fpath, "rb") as f:
        numpy.load(f)
