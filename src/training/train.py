import numpy, random, torch, tqdm

from .. import toolkit

from .SubjectSampleDataMatcher import SubjectSampleDataMatcher
from .DirectComparator import DirectComparator

def main(fpath, repeats, slicelen, batchsize, device):

    data, test = toolkit.save.load(fpath)

    subject_data = next(iter(data.values()))
    sample_data = next(iter(subject_data.values()))
    mfcc = sample_data["mfcc"]
    print(mfcc.shape)
    input()

    matcher = SubjectSampleDataMatcher(
        data,
        repeats = repeats,
        slicelen = slicelen,
        batch_size = batchsize,
        shuffle = True
    )

    tester = SubjectSampleDataMatcher(
        data,
        repeats = repeats,
        slicelen = slicelen,
        batch_size = batchsize*2
    )

    model = DirectComparator(
        layers = [

        ]
    )
    
