import numpy, random, torch, tqdm

from .. import preprocessing, util, modules
from .SubjectSampleDataCreator import SubjectSampleDataCreator
from .SubjectSampleDataMatcher import SubjectSampleDataMatcher
from .Comparator import Comparator
from .DirectComparator import DirectComparator

SEED = 6
SUBJECTS = 7
UTTERANCES = 4
COLUMNS = 3
ROWS = 4
SLICE = 2

def create_dataset(n_subjects=SUBJECTS, n_utterances=UTTERANCES, seed=SEED):

    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    out = {}
    for i in range(n_subjects):
        subject_key = "subject_%d" % i
        sample_data = {}
        out[subject_key] = sample_data
        for j in range(n_utterances):
            sample_key = "sample_%d" % j
            mfcc = numpy.random.rand(ROWS, COLUMNS).T
            mfcc[:,0] = (i/n_subjects)*10-5
            mfcc[:,1] = (j/n_utterances)*2-1
            mfcc[:,2] = numpy.arange(COLUMNS).astype(numpy.float32)/COLUMNS*2-1
            data = {preprocessing.MFCC: mfcc}
            sample_data[sample_key] = data
    return out

class VecComparator(DirectComparator):

    def forward(self, X):
        X = X.float()
        N, C, W, H = X.size()
        return super().forward(X.view(N, C, W*H))

def main():
    
    subjects = 100
    samples = 50

    dataset = create_dataset(subjects, samples)
    testset = create_dataset(subjects, samples, seed=7)

    creator = SubjectSampleDataMatcher(dataset, repeats=4, slicelen=SLICE, batch_size=128, shuffle=True)
    tester = SubjectSampleDataMatcher(testset, repeats=4, slicelen=SLICE, batch_size=128)

    h = 32

    emb_input = SLICE*ROWS

    model = VecComparator(

        layers = [
            torch.nn.Linear(emb_input, h),
            torch.nn.ReLU(),
            torch.nn.Linear(h, h),
        ]
    )

    epochs = 300

    lossf = torch.nn.L1Loss()
    #optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, weight_decay=0)
    optim = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    avg = util.MovingAverage(momentum=0.95)

    with tqdm.tqdm(range(epochs)) as bar:
        for epoch in bar:

            dataloader = creator.create()

            model.train()
            for (X,) in dataloader:
                loss = model.loss(X)
                optim.zero_grad()
                loss.backward()
                optim.step()
                avg.update(loss.item())

            sched.step()

            model.eval()
            Y_sum = train_acc = train_n = acc = n = 0.0
            with torch.no_grad():

                for (X,) in dataloader:
                    train_acc += model.score(X)
                    train_n += len(X)
                
                for (X,) in tester.create():
                    acc += model.score(X)
                    n += len(X)

            acc /= n
            train_acc /= train_n

            bar.set_description("Train (%d) %.3f | Test %.3f | Loss %.4f" % (len(dataloader.dataset), train_acc, acc, avg.peek()))

    print("Final test accuracy: %.4f %%" % (acc*100))
            

            
            
