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
        self.layernorm28 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x295, x302):
        x303=operator.add(x295, x302)
        x304=self.layernorm28(x303)
        return x304

m = M().eval()
x295 = torch.randn(torch.Size([1, 7, 7, 768]))
x302 = torch.randn(torch.Size([1, 7, 7, 768]))
start = time.time()
output = m(x295, x302)
end = time.time()
print(end-start)
