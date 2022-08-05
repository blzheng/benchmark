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
        self.layernorm12 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)

    def forward(self, x111, x125):
        x126=operator.add(x111, x125)
        x127=self.layernorm12(x126)
        return x127

m = M().eval()
x111 = torch.randn(torch.Size([1, 14, 14, 512]))
x125 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x111, x125)
end = time.time()
print(end-start)
