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

    def forward(self, x481):
        x482=torch.cat([x481], 1)
        return x482

m = M().eval()
x481 = torch.randn(torch.Size([1, 896, 7, 7]))
start = time.time()
output = m(x481)
end = time.time()
print(end-start)
