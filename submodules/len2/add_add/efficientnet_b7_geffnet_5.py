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

    def forward(self, x738, x724, x753):
        x739=operator.add(x738, x724)
        x754=operator.add(x753, x739)
        return x754

m = M().eval()
x738 = torch.randn(torch.Size([1, 384, 7, 7]))
x724 = torch.randn(torch.Size([1, 384, 7, 7]))
x753 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x738, x724, x753)
end = time.time()
print(end-start)
