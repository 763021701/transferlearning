# coding=utf-8
import torch
import torch.nn.functional as F

from alg.algs.ERM import ERM


class MMD2(ERM):
    def __init__(self, args):
        super(MMD2, self).__init__(args)
        self.args = args
        self.kernel_type = "gaussian"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches, opt, sch):
        bs = minibatches[0][0].size(0)
        nmb = len(minibatches)

        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        features = self.featurizer(all_x)
        feature_minibatches = torch.split(features, [bs, bs, bs], dim=0)

        objective = 0
        penalty = 0

        all_pred_y = self.classifier(features)
        objective = F.cross_entropy(all_pred_y, all_y)

        for i in range(nmb):
            for j in range(i + 1, nmb):
                penalty += self.mmd(feature_minibatches[i], feature_minibatches[j])

        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        opt.zero_grad()
        (objective + (self.args.mmd_gamma*penalty)).backward()
        opt.step()
        if sch:
            sch.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'class': objective.item(), 'mmd': penalty, 'total': (objective.item() + (self.args.mmd_gamma*penalty))}
