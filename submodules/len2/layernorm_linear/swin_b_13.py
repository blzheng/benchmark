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
        self.layernorm26 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear24 = Linear(in_features=512, out_features=2048, bias=True)

    def forward(self, x287):
        x288=self.layernorm26(x287)
        x289=self.linear24(x288)
        return x289

m = M().eval()
x287 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x287)
end = time.time()
print(end-start)
