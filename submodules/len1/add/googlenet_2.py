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

    def forward(self, x11):
        x12=operator.add(x11, -0.18799999999999994)
        return x12

m = M().eval()
x11 = torch.randn(torch.Size([1, 1, 224, 224]))
start = time.time()
output = m(x11)
end = time.time()
print(end-start)
