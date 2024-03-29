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

    def forward(self, x67):
        x68=torch.nn.functional.relu(x67,inplace=True)
        return x68

m = M().eval()
x67 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x67)
end = time.time()
print(end-start)
