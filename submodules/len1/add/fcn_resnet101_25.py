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

    def forward(self, x270, x262):
        x271=operator.add(x270, x262)
        return x271

m = M().eval()
x270 = torch.randn(torch.Size([1, 1024, 28, 28]))
x262 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x270, x262)
end = time.time()
print(end-start)
