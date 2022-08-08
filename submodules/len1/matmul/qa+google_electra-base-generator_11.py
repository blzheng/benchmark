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

    def forward(self, x264, x252):
        x265=torch.matmul(x264, x252)
        return x265

m = M().eval()
x264 = torch.randn(torch.Size([1, 4, 384, 384]))
x252 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x264, x252)
end = time.time()
print(end-start)
