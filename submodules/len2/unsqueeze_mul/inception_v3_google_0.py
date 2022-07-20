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

    def forward(self, x1):
        x2=torch.unsqueeze(x1, 1)
        x3=operator.mul(x2, 0.458)
        return x3

m = M().eval()
x1 = torch.randn(torch.Size([1, 224, 224]))
start = time.time()
output = m(x1)
end = time.time()
print(end-start)
