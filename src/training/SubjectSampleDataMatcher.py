import torch
import torch.utils.data

from .SubjectSampleDataCreator import SubjectSampleDataCreator

class SubjectSampleDataMatcher(SubjectSampleDataCreator):

    def create(self):
        Xs = [self.construct_data() for i in range(self.n)]
        X = torch.cat(Xs, dim=0).float()
        tensorset = torch.utils.data.TensorDataset(X)
        return torch.utils.data.DataLoader(tensorset, **self.kwargs)

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
