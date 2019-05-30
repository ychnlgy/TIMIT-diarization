import os

from .preprocess import preprocess

from .. import toolkit

MFCC_EXT = "-mfcc"

def save(savepath):
    fpath, ext = os.path.splitext(savepath)
    outpath = fpath + MFCC_EXT + ext
    toolkit.save.save(list(preprocess(savepath)), outpath)
