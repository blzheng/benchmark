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
        self.layernorm34 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)

    def forward(self, x364, x378):
        x379=operator.add(x364, x378)
        x380=self.layernorm34(x379)
        return x380

m = M().eval()
x364 = torch.randn(torch.Size([1, 14, 14, 512]))
x378 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x364, x378)
end = time.time()
print(end-start)
