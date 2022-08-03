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

    def forward(self, x185):
        x187=operator.getitem(x185, 1)
        return x187

m = M().eval()
x185 = (torch.randn((torch.Size([1, 244, 14, 14]), torch.randn(torch.Size([1, 244, 14, 14]), )
start = time.time()
output = m(x185)
end = time.time()
print(end-start)
