import torch
import torch.utils.data

from .SubjectSampleDataCreator import SubjectSampleDataCreator

class SubjectSampleDataMatcher(SubjectSampleDataCreator):

    def create(self, get_stats=False, miu=None, std=None):
        Xs = [self.construct_data() for i in range(self.n)]
        X = torch.cat(Xs, dim=0).float()
        if miu is not None:
            X = (X-miu)/std
        tensorset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(tensorset, **self.kwargs)
        if get_stats:
            return (
                loader,
                X.mean(dim=1).unsqueeze(1),
                X.std(dim=1).unsqueeze(1)
            )
        else:
            return loader

    # === PROTECTED ===

    def construct_data(self):
        pro, ant = self._halve_by_subject()
        pr2, an2 = self._reverse_roles(pro, ant)
        X1 = self.match_by_sample(pro, ant)
        X2 = self.match_by_sample(pr2, an2)
        return torch.cat([X1, X2], dim=0)

    def match_by_sample(self, pro, ant):
        assert len(ant) >= len(pro)
        ant = ant[:len(pro)]
        sam = self._flatten_after_same_subject(pro)
        ant = self._flatten_after_sampling(pro, ant)
        pro = self._flatten_by_sample(pro)
        return torch.stack([pro, sam, ant], dim=1)
