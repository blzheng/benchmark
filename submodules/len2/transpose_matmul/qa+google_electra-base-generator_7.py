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

    def forward(self, x330, x341):
        x342=x330.transpose(-1, -2)
        x343=torch.matmul(x341, x342)
        return x343

m = M().eval()
x330 = torch.randn(torch.Size([1, 4, 384, 64]))
x341 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x330, x341)
end = time.time()
print(end-start)
