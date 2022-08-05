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

    def forward(self, x527, x528, x529, x530):
        x531=torch.cat([x527, x528, x529, x530], -1)
        return x531

m = M().eval()
x527 = torch.randn(torch.Size([1, 7, 7, 512]))
x528 = torch.randn(torch.Size([1, 7, 7, 512]))
x529 = torch.randn(torch.Size([1, 7, 7, 512]))
x530 = torch.randn(torch.Size([1, 7, 7, 512]))
start = time.time()
output = m(x527, x528, x529, x530)
end = time.time()
print(end-start)
