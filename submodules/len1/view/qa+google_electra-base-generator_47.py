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

    def forward(self, x519, x522):
        x523=x519.view(x522)
        return x523

m = M().eval()
x519 = torch.randn(torch.Size([1, 384, 4, 64]))
x522 = (1, 384, 256, )
start = time.time()
output = m(x519, x522)
end = time.time()
print(end-start)
