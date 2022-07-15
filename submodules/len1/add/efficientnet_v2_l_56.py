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

    def forward(self, x814, x799):
        x815=operator.add(x814, x799)
        return x815

m = M().eval()
x814 = torch.randn(torch.Size([1, 384, 7, 7]))
x799 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x814, x799)
end = time.time()
print(end-start)
