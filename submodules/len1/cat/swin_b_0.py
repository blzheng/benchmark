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

    def forward(self, x51, x52, x53, x54):
        x55=torch.cat([x51, x52, x53, x54], -1)
        return x55

m = M().eval()
x51 = torch.randn(torch.Size([1, 28, 28, 128]))
x52 = torch.randn(torch.Size([1, 28, 28, 128]))
x53 = torch.randn(torch.Size([1, 28, 28, 128]))
x54 = torch.randn(torch.Size([1, 28, 28, 128]))
start = time.time()
output = m(x51, x52, x53, x54)
end = time.time()
print(end-start)
