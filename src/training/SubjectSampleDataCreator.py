import torch, random
import torch.utils.data

from .. import preprocessing, toolkit

class SubjectSampleDataCreator:

    def __init__(self, data, repeats, slicelen, **kwargs):
        self.data = data
        self.n = repeats
        self.slclen = slicelen
        self._samples = self._get_samples()
        self.kwargs = kwargs

    def create(self):
        Xs, Ys = [], []
        for i in range(self.n):
            X, Y = self.construct_data()
            Xs.append(X)
            Ys.append(Y)
        X = torch.cat(Xs, dim=0)
        Y = torch.cat(Ys, dim=0)
        tensorset  = torch.utils.data.TensorDataset(X, Y)
        return torch.utils.data.DataLoader(tensorset, **self.kwargs)

    # === PROTECTED ===

    def select_x(self, sample):
        x = sample[preprocessing.MFCC].T
        x = torch.from_numpy(x)
        l = x.size(1)
        assert l >= self.slclen
        i = random.randint(0, l-self.slclen)
        j = i + self.slclen
        return x[:,i:j]

    def construct_data(self):
        pro, ant = self._halve_by_subject()
        pr2, an2 = self._reverse_roles(pro, ant)
        X1, Y1 = self.match_by_sample(pro, ant)
        X2, Y2 = self.match_by_sample(pr2, an2)
        return self._join(X1, Y1, X2, Y2)

    def match_by_sample(self, pro, ant):
        assert len(ant) >= len(pro)
        sam = self._flatten_after_same_subject(pro)
        ant = self._flatten_after_sampling(pro, ant)
        pro = self._flatten_by_sample(pro)
        pos = torch.stack([pro, sam], dim=1)
        neg = torch.stack([pro, ant], dim=1)
        y_p = torch.ones(len(pos))
        y_n = torch.ones(len(neg)) * -1
        X = torch.cat([pos, neg], dim=0)
        Y = torch.cat([y_p, y_n], dim=0)
        return X, Y

    # === PRIVATE ===

    def _get_samples(self):
        subject_data = next(iter(self.data.values()))
        return list(subject_data.keys())

    def _halve_by_subject(self):
        subjects = list(self.data.values())
        random.shuffle(subjects)
        N = len(subjects)//2
        return subjects[:N], subjects[N:]

    def _reverse_roles(self, pro, ant):
        N = min(len(pro), len(ant))
        random.shuffle(ant)
        random.shuffle(pro)
        ant = ant[:N]
        pro = pro[:N]
        return ant, pro

    def _join(self, X1, Y1, X2, Y2):
        X = torch.cat([X1, X2], dim=0) # (N, 2, NUMCEP, self.slicelen)
        Y = torch.cat([Y1, Y2], dim=0) # (N)
        return X, Y

    @toolkit.torchtools.torchstack
    def _flatten_by_sample(self, out):
        for subject_data in out:
            for sample_id in self._samples:
                yield self.select_x(subject_data[sample_id])

    @toolkit.torchtools.torchstack
    def _flatten_after_sampling(self, pro, ant):
        n = len(ant)-1
        for _ in pro:
            for sample_id in self._samples:
                yield self.select_x(ant[random.randint(0, n)][sample_id])

    @toolkit.torchtools.torchstack
    def _flatten_after_same_subject(self, pro):
        n = len(self._samples)-1
        for subject_data in pro:
            print(subject_data.keys())
            print(self._samples)
            input()
            for _ in self._samples:
                sample_id = self._samples[random.randint(0, n)]
                yield self.select_x(subject_data[sample_id])

    @staticmethod
    def test():

        # Tests

        import numpy

        from . import toy 

        dataset = toy.create_dataset()

        creator = SubjectSampleDataCreator(dataset, repeats=1, slicelen=2, batch_size=1)

        dataloader = creator.create()

        set1 = list(dataloader)
        set2 = list(creator.create())

        def same_float(f1, f2, eps=1e-8):
            return abs(f1 - f2) < eps

        def same_tensor(t1, t2, eps=1e-3):
            return (t1-t2).norm() < eps

        class Classifier:

            def predict(self, X):
                assert X.size(1) == 2
                x1 = X[:,0]
                x2 = X[:,1]

                s1 = self.get_subject(x1)
                s2 = self.get_subject(x2)

                if same_float(s1, s2):
                    return 1
                else:
                    return -1

            def get_subject(self, x):
                assert x.size(0) == 1
                x = x.squeeze(0)
                subject_id = x[0]
                sample_id = x[1]
                column_id = x[2]
                data = x[3]

                assert same_float(subject_id[0], subject_id[1])
                assert subject_id[0] < toy.SUBJECTS
                assert same_float(sample_id[0], sample_id[1])
                assert sample_id[0] < toy.UTTERANCES
                assert column_id[0] < toy.COLUMNS and column_id[1] < toy.COLUMNS
                assert same_float(column_id[0] + 1, column_id[1])

                return subject_id[0]

        assert same_tensor(
            set1[0][0],
            torch.DoubleTensor([
                [[[2.0000, 2.0000],
                  [0.0000, 0.0000],
                  [1.0000, 2.0000],
                  [0.1794, 0.7833]],

                 [[2.0000, 2.0000],
                  [2.0000, 2.0000],
                  [0.0000, 1.0000],
                  [0.0116, 0.2767]]]
            ])
        )

        assert same_tensor(
            set1[-1][0],
            torch.DoubleTensor([
                [[[6.0000, 6.0000],
                  [3.0000, 3.0000],
                  [0.0000, 1.0000],
                  [0.3050, 0.9847]],

                 [[1.0000, 1.0000],
                  [3.0000, 3.0000],
                  [0.0000, 1.0000],
                  [0.4904, 0.0366]]]
            ])
        )

        assert same_tensor(
            set2[-1][0],
            torch.DoubleTensor([
                [[[4.0000, 4.0000],
                  [3.0000, 3.0000],
                  [0.0000, 1.0000],
                  [0.6355, 0.7244]],

                 [[1.0000, 1.0000],
                  [3.0000, 3.0000],
                  [0.0000, 1.0000],
                  [0.4904, 0.0366]]]
            ])
        )

        classifier = Classifier()

        for x, y in set1:
            acc = int(same_float(y, classifier.predict(x)))
            assert same_float(acc, 1)

        print("All tests completed.")
                
                
