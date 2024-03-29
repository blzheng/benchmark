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

    def forward(self, x766, x751):
        x767=operator.add(x766, x751)
        return x767

m = M().eval()
x766 = torch.randn(torch.Size([1, 384, 7, 7]))
x751 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x766, x751)
end = time.time()
print(end-start)
