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
        self.layernorm4 = LayerNorm((384,), eps=1e-06, elementwise_affine=True)

    def forward(self, x59):
        x60=self.layernorm4(x59)
        return x60

m = M().eval()
x59 = torch.randn(torch.Size([1, 28, 28, 384]))
start = time.time()
output = m(x59)
end = time.time()
print(end-start)
