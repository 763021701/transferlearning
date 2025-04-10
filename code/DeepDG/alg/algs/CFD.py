# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.algs.ERM import ERM


class SampleNet(nn.Module):
    """
    TNet module for adversarial networks with fixed activation layers and predefined parameters.
    """

    def __init__(self, feature_dim=64, t_batchsize=64, t_var=1):
        super(SampleNet, self).__init__()
        self.feature_dim = feature_dim  # Feature dimension
        self.t_sigma_num = t_batchsize // 16  # Number of sigmas for t_net
        self._input_adv_t_net_dim = feature_dim  # Input noise dimension
        self._input_t_dim = feature_dim  # t_net input dimension
        self._input_t_batchsize = t_batchsize  # Batch size
        self._input_t_var = t_var  # Variance of input noise

        # Fixed activation layers
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_2 = nn.Tanh()

        # Create a simple 3-layer fully connected network using fixed activation layers
        self.t_layers_list = nn.ModuleList()
        ch_in = self.feature_dim
        num_layer = 3
        for i in range(num_layer):
            self.t_layers_list.append(nn.Linear(ch_in, ch_in))
            self.t_layers_list.append(nn.BatchNorm1d(ch_in))
            # Use activation_1 for the first two layers, and activation_2 for the last layer
            self.t_layers_list.append(
                self.activation_1 if i < (num_layer - 1) else self.activation_2
            )

    def forward(self):
        device = next(self.t_layers_list.parameters()).device
        # Generate white noise
        if self.t_sigma_num > 0:
            # Initialize the white noise input
            self._t_net_input = torch.randn(
                self.t_sigma_num, self._input_adv_t_net_dim
            ) * (self._input_t_var**0.5)
            self._t_net_input = self._t_net_input.to(device).detach()

            # Forward pass
            a = self._t_net_input
            for layer in self.t_layers_list:
                a = layer(a)

            a = a.repeat(int(self._input_t_batchsize / self.t_sigma_num), 1)

            # Generate the final t value
            # self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * ((self._input_t_var / self._input_t_dim) ** 0.5)
            # self._t = self._t.to(device).detach()
            self._t = a
        else:
            # When t_sigma_num = 0, generate standard Gaussian noise as t
            self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * (
                (self._input_t_var / self._input_t_dim) ** 0.5
            )
            self._t = self._t.to(device).detach()
        return self._t


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)


class CFLossFunc(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference.
    Args:
        alpha_for_loss: the weight for amplitude in CF loss, from 0-1
        beta_for_loss: the weight for phase in CF loss, from 0-1
    """

    def __init__(self, alpha_for_loss=0.5, beta_for_loss=0.5, args=None):
        super(CFLossFunc, self).__init__()
        self.initial_alpha = alpha_for_loss
        self.alpha = alpha_for_loss
        self.beta = beta_for_loss
        if args is not None:
            self.max_epoch = args.max_epoch

    def update_alpha_beta(self, epoch):
        if epoch < self.max_epoch:
            self.alpha = self.initial_alpha * (1 - epoch / self.max_epoch)
        else:
            self.alpha = 0

        self.beta = 1 - self.alpha

    def forward(self, feat_tg, feat, t=None, args=None, epoch=0):
        """
        Calculate CF loss between target and synthetic features.
        Args:
            feat_tg: target features from real data [B1 x D]
            feat: synthetic features [B2 x D]
            args: additional arguments containing num_freqs
        """
        # Generate random frequencies
        if t is None:
            t = torch.randn((args.cfd_t_batchsize, feat.size(1)), device=feat.device)
        t_x_real = calculate_real(torch.matmul(t, feat.t()))
        t_x_imag = calculate_imag(torch.matmul(t, feat.t()))
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target_real = calculate_real(torch.matmul(t, feat_tg.t()))
        t_target_imag = calculate_imag(torch.matmul(t, feat_tg.t()))
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        # Calculate amplitude difference and phase difference
        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (
            torch.mul(t_target_norm, t_x_norm)
            - torch.mul(t_x_real, t_target_real)
            - torch.mul(t_x_imag, t_target_imag)
        )

        loss_pha = loss_pha.clamp(min=1e-12)  # Ensure numerical stability

        self.update_alpha_beta(epoch)

        # Combine losses
        loss = torch.mean(torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha))
        return loss


net_dim_dict = {"resnet18": 512, "resnet50": 2048, "vgg16": 1000}

class CFD(ERM):
    def __init__(self, args):
        super(CFD, self).__init__(args)
        self.args = args
        self.cf_loss = CFLossFunc(self.args.cfd_alpha, self.args.cfd_beta, args)
        self.sample_net = SampleNet(net_dim_dict[args.net], self.args.cfd_t_batchsize, 1)
        self.sample_net_opt = torch.optim.AdamW(
            self.sample_net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

    def cfd(self, x, y, t, epoch):
        return self.cf_loss(x, y, t, self.args, epoch)

    def update(self, minibatches, opt, sch, epoch):
        nmb = len(minibatches)

        # train sample net
        self.sample_net.train()
        t = self.sample_net()

        features = [self.featurizer(
            data[0].cuda().float()) for data in minibatches]

        penalty_t = 0

        for i in range(nmb):
            for j in range(i + 1, nmb):
                feat = F.normalize(features[i].detach(), dim=1)
                feat_tg = F.normalize(features[j].detach(), dim=1)
                penalty_t -= self.cfd(feat_tg, feat, t, epoch)

        if nmb > 1:
            penalty_t /= (nmb * (nmb - 1) / 2)

        self.sample_net_opt.zero_grad()
        penalty_t.backward()
        self.sample_net_opt.step()

        # train main net
        classifs = [self.classifier(fi) for fi in features]
        targets = [data[1].cuda().long() for data in minibatches]

        objective = 0
        penalty = 0

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                feat = F.normalize(features[i], dim=1)
                feat_tg = F.normalize(features[j], dim=1)
                penalty += self.cfd(feat_tg, feat, t.detach(), epoch)

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
