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

    def forward(self, x278, x270):
        x279=operator.add(x278, x270)
        return x279

m = M().eval()
x278 = torch.randn(torch.Size([1, 1024, 14, 14]))
x270 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x278, x270)
end = time.time()
print(end-start)
