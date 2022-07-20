import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.layernorm22 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)
        self.linear44 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu22 = GELU(approximate='none')
        self.linear45 = Linear(in_features=1536, out_features=384, bias=True)

    def forward(self, x263):
        x264=self.layernorm22(x263)
        x265=self.linear44(x264)
        x266=self.gelu22(x265)
        x267=self.linear45(x266)
        return x267

m = M().eval()
x263 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x263)
end = time.time()
print(end-start)
