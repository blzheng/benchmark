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

    def forward(self, x263, x251):
        x264=torch.matmul(x263, x251)
        return x264

m = M().eval()
x263 = torch.randn(torch.Size([1, 12, 384, 384]))
x251 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x263, x251)
end = time.time()
print(end-start)
