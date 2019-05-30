def save(obj, fpath):
    with open(fpath, "wb") as f:
        numpy.save(f, obj)

def load(fpath):
    with open(savepath, "rb") as f:
        numpy.load(f)
