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
        self.layernorm12 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.linear24 = Linear(in_features=512, out_features=2048, bias=True)

    def forward(self, x153):
        x154=self.layernorm12(x153)
        x155=self.linear24(x154)
        return x155

m = M().eval()
x153 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
