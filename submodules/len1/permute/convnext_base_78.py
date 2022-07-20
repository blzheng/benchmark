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

    def forward(self, x411):
        x412=torch.permute(x411, [0, 2, 3, 1])
        return x412

m = M().eval()
x411 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x411)
end = time.time()
print(end-start)
