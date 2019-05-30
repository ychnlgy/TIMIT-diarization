from . import load

from .. import toolkit

def save(datadir, savepath):
    data, test = load.load(datadir)
    toolkit.save.save([data, test], savepath)
