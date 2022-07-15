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

    def forward(self, x1068, x1053):
        x1069=operator.add(x1068, x1053)
        return x1069

m = M().eval()
x1068 = torch.randn(torch.Size([1, 640, 7, 7]))
x1053 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x1068, x1053)
end = time.time()
print(end-start)
