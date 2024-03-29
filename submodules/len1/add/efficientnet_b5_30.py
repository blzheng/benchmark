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

    def forward(self, x587, x572):
        x588=operator.add(x587, x572)
        return x588

m = M().eval()
x587 = torch.randn(torch.Size([1, 512, 7, 7]))
x572 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x587, x572)
end = time.time()
print(end-start)
