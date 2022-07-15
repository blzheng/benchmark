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

    def forward(self, x826, x821):
        x827=operator.mul(x826, x821)
        return x827

m = M().eval()
x826 = torch.randn(torch.Size([1, 2304, 1, 1]))
x821 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x826, x821)
end = time.time()
print(end-start)
