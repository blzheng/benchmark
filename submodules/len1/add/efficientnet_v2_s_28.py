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

    def forward(self, x442, x427):
        x443=operator.add(x442, x427)
        return x443

m = M().eval()
x442 = torch.randn(torch.Size([1, 256, 7, 7]))
x427 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x442, x427)
end = time.time()
print(end-start)
