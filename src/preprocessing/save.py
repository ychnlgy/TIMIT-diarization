import os

from .preprocess import preprocess

from .. import toolkit

MFCC_EXT = "-mfcc"

def save(savepath, keepwave):
    fpath, ext = os.path.splitext(savepath)
    outpath = fpath + MFCC_EXT + ext
    toolkit.save.save(list(preprocess(savepath, keepwave)), outpath)
