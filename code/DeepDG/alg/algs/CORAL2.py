# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM


class CORAL2(ERM):
    def __init__(self, args):
        super(CORAL2, self).__init__(args)
        self.args = args
        self.kernel_type = "mean_cov"

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

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
                penalty += self.coral(feature_minibatches[i], feature_minibatches[j])

        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        opt.zero_grad()
        (objective + (self.args.mmd_gamma*penalty)).backward()
        opt.step()
        if sch:
            sch.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'class': objective.item(), 'coral': penalty, 'total': (objective.item() + (self.args.mmd_gamma*penalty))}
