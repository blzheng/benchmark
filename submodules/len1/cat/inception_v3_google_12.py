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

    def forward(self, x264, x274, x287, x291):
        x292=torch.cat([x264, x274, x287, x291], 1)
        return x292

m = M().eval()
x264 = torch.randn(torch.Size([1, 320, 5, 5]))
x274 = torch.randn(torch.Size([1, 768, 5, 5]))
x287 = torch.randn(torch.Size([1, 768, 5, 5]))
x291 = torch.randn(torch.Size([1, 192, 5, 5]))
start = time.time()
output = m(x264, x274, x287, x291)
end = time.time()
print(end-start)
