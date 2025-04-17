import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleNet(nn.Module):
    """
    SampleNet module for generating sample points for CFD loss computation
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

class CFLoss(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference.
    """
    def __init__(self, alpha=0.5, beta=0.5, t_batchsize=64, feature_dim=512, max_iter=1000, **kwargs):
        super(CFLoss, self).__init__()
        self.initial_alpha = alpha
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.sample_net = SampleNet(feature_dim=feature_dim, t_batchsize=t_batchsize)
        self.sample_optim = torch.optim.AdamW(self.sample_net.parameters(), lr=0.001, weight_decay=5e-4)
        self.cur_iter = 0

    def update_alpha_beta(self, epoch):
        if epoch < self.max_iter:
            self.alpha = self.initial_alpha * (1 - epoch / self.max_iter)
        else:
            self.alpha = 0

        self.beta = 1 - self.alpha

    def _compute_loss(self, feat_tg, feat, t=None):
        """
        Calculate CF loss between target and source features.
        """
        if t is None:
            t = self.sample_net()

        # Normalize features
        feat = F.normalize(feat, dim=1)
        feat_tg = F.normalize(feat_tg, dim=1)

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

        # Combine losses
        loss = torch.mean(torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha))
        return loss

    def train_sample_net(self, source, target):
        """Train the sample network with adversarial loss"""
        self.sample_net.train()
        t = self.sample_net()

        # We want to maximize the CF loss for better sampling
        loss = -self._compute_loss(target, source, t)

        self.sample_optim.zero_grad()
        loss.backward()
        self.sample_optim.step()

        self.sample_net.eval()
        t = self.sample_net()

        return t.detach()

    def forward(self, source, target, epoch=None):
        """
        Forward pass for computing CFD loss
        Args:
            source: source domain features
            target: target domain features
            epoch: current training epoch or iteration
        """
        if epoch is not None:
            self.update_alpha_beta(epoch)

        # Train sample net and get the sampling points
        t = self.train_sample_net(source.detach(), target.detach())

        # Compute actual CFD loss
        loss = self._compute_loss(target, source, t)

        self.cur_iter += 1
        return loss
