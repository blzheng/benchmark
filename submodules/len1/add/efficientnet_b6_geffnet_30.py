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

    def forward(self, x545, x531):
        x546=operator.add(x545, x531)
        return x546

m = M().eval()
x545 = torch.randn(torch.Size([1, 344, 7, 7]))
x531 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x545, x531)
end = time.time()
print(end-start)
