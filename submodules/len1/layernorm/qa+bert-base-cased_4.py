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
        self.layernorm4 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x111):
        x112=self.layernorm4(x111)
        return x112

m = M().eval()
x111 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x111)
end = time.time()
print(end-start)
