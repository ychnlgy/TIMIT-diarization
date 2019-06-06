import numpy, random, torch, tqdm, sys

from .. import toolkit, preprocessing, modules, util

from .SubjectSampleDataMatcher import SubjectSampleDataMatcher
from .DirectComparator import DirectComparator

def report_results(results_path, msg):
    with open(results_path, "a") as results_file:
        results_file.write(msg)

def main(fpath, repeats, slicelen, batchsize, l2reg, device, results_path="results.txt", save_cycle=5, save_path="model.pkl"):

    log = util.Log(results_path)
    log.write(" ".join(sys.argv))

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

    assert slicelen == preprocessing.NUMCEP == 16

    create_act = lambda c: torch.nn.ReLU()

    channels = [16, 32, 64]
    latent_features = 32

    model = DirectComparator(
        layers = [
            modules.Reshape(1, 16, 16),
            torch.nn.Conv2d(1, 16, 3, padding=1),
            modules.ResNet(
                block_constructor = lambda in_c, out_c: torch.nn.Sequential(
                    create_act(in_c),
                    torch.nn.Conv2d(in_c, out_c, 3, padding=1, stride=int(out_c>in_c)+1, bias=False),
                    torch.nn.BatchNorm2d(out_c),

                    create_act(out_c),
                    torch.nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(out_c)
                ),
                shortcut_constructor = lambda in_c, out_c: modules.Shortcut(
                    in_c, out_c, stride=2, act=create_act(out_c)
                ),
                channels = channels,
                block_depth = 2
            ),
            create_act(channels[-1]),
            torch.nn.AvgPool2d(4),
            modules.Reshape(channels[-1]),
            torch.nn.Linear(channels[-1], latent_features)
        ]
    ).to(device)

    parameters = sum(torch.numel(p) for p in model.parameters() if p.requires_grad)

    print("Parameters: %d" % parameters)

    data_creator = SubjectSampleDataMatcher(data, repeats, slicelen, batch_size=batchsize, shuffle=True)
    test_creator = SubjectSampleDataMatcher(test, repeats, slicelen, batch_size=batchsize*2)

    epochs = 300

    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=l2reg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    avg = util.MovingAverage(momentum=0.99)

    for epoch in range(0, epochs+1):

        model.train()

        #dataset, miu, std = data_creator.create(get_stats=True)
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
                data_n += len(X)

            for (X,) in test_creator.create():#test_creator.create(miu=miu, std=std):
                test_a += model.score(X.to(device))
                test_n += len(X)

            data_acc = data_a / data_n * 100
            test_acc = test_a / test_n * 100

            log.write(
                msg = "Epoch %d training|test accuracy: %.2f|%.2f" % (epoch, data_acc, test_acc)
            )

        if not epoch % save_cycle:
            torch.save(model.state_dict(), save_path)
            log.write(msg = "Saved model to %s" % save_path)
                
    
    
    
