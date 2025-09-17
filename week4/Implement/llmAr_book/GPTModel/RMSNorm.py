import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x.norm(p=2, dim=-1, keepdim=True) # 计算L2范数
        rms_x = norm_x / (x.size(-1) ** 0.5) # 计算RMS值
        x = x / (rms_x + self.eps)
        return x * self.scale