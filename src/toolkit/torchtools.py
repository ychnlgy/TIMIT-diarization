import torch

def torchstack(it_fn):

    def _wrapped(*args, **kwargs):
        it = it_fn(*args, **kwargs)
        return torch.stack(list(it), dim=0)

    return _wrapped
    
