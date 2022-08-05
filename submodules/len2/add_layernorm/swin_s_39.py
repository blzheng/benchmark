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
        self.layernorm45 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)

    def forward(self, x494, x501):
        x502=operator.add(x494, x501)
        x503=self.layernorm45(x502)
        return x503

m = M().eval()
x494 = torch.randn(torch.Size([1, 14, 14, 384]))
x501 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x494, x501)
end = time.time()
print(end-start)
