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
        self.layernorm2 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear7 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x69, x65):
        x70=operator.add(x69, x65)
        x71=self.layernorm2(x70)
        x72=self.linear7(x71)
        return x72

m = M().eval()
x69 = torch.randn(torch.Size([1, 384, 256]))
x65 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x69, x65)
end = time.time()
print(end-start)
