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
        self.layernorm24 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear22 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu10 = GELU(approximate='none')

    def forward(self, x249, x263):
        x264=operator.add(x249, x263)
        x265=self.layernorm24(x264)
        x266=self.linear22(x265)
        x267=self.gelu10(x266)
        return x267

m = M().eval()
x249 = torch.randn(torch.Size([1, 14, 14, 384]))
x263 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x249, x263)
end = time.time()
print(end-start)
