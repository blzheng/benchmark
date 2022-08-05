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

    def forward(self, x264):
        x265=self.layernorm24(x264)
        x266=self.linear22(x265)
        return x266

m = M().eval()
x264 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x264)
end = time.time()
print(end-start)
