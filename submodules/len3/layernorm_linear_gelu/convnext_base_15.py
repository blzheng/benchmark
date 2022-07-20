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
        self.layernorm15 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.linear30 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu15 = GELU(approximate='none')

    def forward(self, x186):
        x187=self.layernorm15(x186)
        x188=self.linear30(x187)
        x189=self.gelu15(x188)
        return x189

m = M().eval()
x186 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x186)
end = time.time()
print(end-start)
