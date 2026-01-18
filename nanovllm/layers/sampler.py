import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    # 从模型输出的 Logits 中选择最终的 Token
    # - Argmax: 贪婪解码（Greedy Decoding），直接选概率最大的。
    # - Top-K / Top-P: 核采样（Nucleus Sampling），增加生成的多样性。
    # - Temperature: 调整概率分布的平滑程度。
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
