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
        self.layernorm8 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.linear16 = Linear(in_features=512, out_features=2048, bias=True)

    def forward(self, x109):
        x110=self.layernorm8(x109)
        x111=self.linear16(x110)
        return x111

m = M().eval()
x109 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
