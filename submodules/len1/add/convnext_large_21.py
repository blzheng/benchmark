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

    def forward(self, x259, x249):
        x260=operator.add(x259, x249)
        return x260

m = M().eval()
x259 = torch.randn(torch.Size([1, 768, 14, 14]))
x249 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x259, x249)
end = time.time()
print(end-start)
