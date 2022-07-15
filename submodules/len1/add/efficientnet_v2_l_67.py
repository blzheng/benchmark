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

    def forward(self, x1004, x989):
        x1005=operator.add(x1004, x989)
        return x1005

m = M().eval()
x1004 = torch.randn(torch.Size([1, 640, 7, 7]))
x989 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x1004, x989)
end = time.time()
print(end-start)
