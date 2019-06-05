import numpy, random, torch, tqdm, sys

from .. import toolkit, preprocessing, modules, util

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
                        torch.nn.BatchNorm2d(out_c),

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

    print("Parameters: %d" % parameters)

    print(model)
    input()

    data_creator = SubjectSampleDataMatcher(data, repeats, slicelen, batch_size=batchsize, shuffle=True)
    test_creator = SubjectSampleDataMatcher(test, repeats, slicelen, batch_size=batchsize*2)

    epochs = 300

    optim = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    avg = util.MovingAverage(momentum=0.99)

    for epoch in range(epochs):

        model.train()

        dataset = data_creator.create()

        with tqdm.tqdm(dataset, ncols=80) as bar:
            for (X,) in bar:
                loss = model.loss(X.to(device))
                optim.zero_grad()
                loss.backward()
                optim.step()
                avg.update(loss.item())
                bar.set_description("[E%d] Loss %.4f" % (epoch, avg.peek()))

        sched.step()

        model.eval()

        with torch.no_grad():

            data_a = data_n = test_a = test_n = 0

            for (X,) in dataset:
                data_a += model.score(X.to(device))
                data_n += len(data_a)

            for (X,) in test_creator.create():
                test_a += model.score(X.to(device))
                test_n += len(test_a)

            data_acc = data_a / data_n * 100
            test_acc = test_a / test_n * 100

            sys.stderr.write("Training | test accuracy: %.2f | %.2f\n" % (data_acc, test_acc))
    
    
    
