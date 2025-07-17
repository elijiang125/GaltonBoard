import torch


class JSDistance(torch.nn.Module):
    """
    Adapted form: https://discuss.pytorch.org/t/jensen-shannon-divergence/2626
    Computes Jensen-Shannon distance metric to be used as a loss function for a Neural Network.
    """
    def __init__(self):
        super(JSDistance, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p = p.view(-1, p.size(-1))
        q = q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()  # Mixture distribution

        js_div = 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))

        return js_div ** 0.5


class Wasserstein(torch.nn.Module):
    """
    Computes Wasserstein (earth mover's) distance
    """
    pass
