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

    def forward(self, x790, x785):
        x791=operator.mul(x790, x785)
        return x791

m = M().eval()
x790 = torch.randn(torch.Size([1, 2304, 1, 1]))
x785 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x790, x785)
end = time.time()
print(end-start)
