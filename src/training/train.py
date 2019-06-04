import numpy, random, torch, tqdm

from .. import toolkit, preprocessing, modules

from .SubjectSampleDataMatcher import SubjectSampleDataMatcher
from .DirectComparator import DirectComparator

def main(fpath, repeats, slicelen, batchsize, device):

    data, test = toolkit.save.load(fpath)

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

    assert slicelen == 64 and preprocessing.NUMCEP == 64

    create_act = lambda c: torch.nn.ReLU()

    channels = [16, 32, 64, 128]
    latent_features = 32

    model = DirectComparator(
        layers = [
            modules.Reshape(1, 64, 64),
            torch.nn.Conv2d(1, 16, 3, padding=1),
            modules.ResNet(
                block_constructor = lambda in_c, out_c: modules.ShakeBlock(
                    lambda: torch.nn.Sequential(
                        create_act(in_c),
                        torch.nn.Conv2d(in_c, out_c, 3, padding=1, stride=2, bias=False),
                        torch.BatchNorm2d(out_c),

                        create_act(out_c),
                        torch.nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                        torch.nn.BatchNorm2d(out_c)
                    )
                ),
                shortcut_constructor = lambda in_c, out_c: modules.Shortcut(
                    in_c, out_c, stride=2, act = create_act(out_c)
                ),
                channels = channels,
                block_depth = 2
            ),
            create_act(channels[-1]),
            torch.nn.AvgPool2d(8),
            modules.Reshape(channels[-1]),
            torch.nn.Linear(channels[-1], latent_features)
        ]
    ).to(device)

    parameters = sum(torch.numel(p) for p in model.parameters() if p.requires_grad)
    
    
