# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.algs.ERM import ERM


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))

def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)

def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)

class CFLossFunc(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference
    Args:
        alpha: the weight for amplitude in CF loss, from 0-1
        beta: the weight for phase in CF loss, from 0-1

    """

    def __init__(self, alpha=1, beta=1):
        super(CFLossFunc, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, target):
        # t_x = torch.mm(t, x.t())
        t_x_real = calculate_real(x.t())
        t_x_imag = calculate_imag(x.t())
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        # t_target = torch.mm(t, target.t())
        t_target_real = calculate_real(target.t())
        t_target_imag = calculate_imag(target.t())
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (torch.mul(t_target_norm, t_x_norm) -
                        torch.mul(t_x_real, t_target_real) -
                        torch.mul(t_x_imag, t_target_imag))

        loss_pha = loss_pha.clamp(min=1e-12)  # keep numerical stability


        loss = torch.sqrt(torch.mean(self.alpha * loss_amp + self.beta * loss_pha))
        return loss


class CFD(ERM):
    def __init__(self, args):
        super(CFD, self).__init__(args)
        self.args = args
        self.cf_loss = CFLossFunc(alpha=1, beta=1)
    
    def cfd(self, x, y):
        self.cf_loss(x, y)

    def update(self, minibatches, opt, sch):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(
            data[0].cuda().float()) for data in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [data[1].cuda().long() for data in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.cfd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        opt.zero_grad()
        (objective + (self.args.cfd_gamma*penalty)).backward()
        opt.step()
        if sch:
            sch.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'class': objective.item(), 'cfd': penalty, 'total': (objective.item() + (self.args.cfd_gamma*penalty))}
