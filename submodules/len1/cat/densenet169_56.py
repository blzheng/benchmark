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

    def forward(self, x369, x376, x383, x390):
        x391=torch.cat([x369, x376, x383, x390], 1)
        return x391

m = M().eval()
x369 = torch.randn(torch.Size([1, 640, 7, 7]))
x376 = torch.randn(torch.Size([1, 32, 7, 7]))
x383 = torch.randn(torch.Size([1, 32, 7, 7]))
x390 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x369, x376, x383, x390)
end = time.time()
print(end-start)
