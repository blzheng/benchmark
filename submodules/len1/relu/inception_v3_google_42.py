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

    def forward(self, x153):
        x154=torch.nn.functional.relu(x153,inplace=True)
        return x154

m = M().eval()
x153 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x153)
end = time.time()
print(end-start)
