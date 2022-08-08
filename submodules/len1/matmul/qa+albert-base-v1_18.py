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

    def forward(self, x370, x381):
        x382=torch.matmul(x370, x381)
        return x382

m = M().eval()
x370 = torch.randn(torch.Size([1, 12, 384, 64]))
x381 = torch.randn(torch.Size([1, 12, 64, 384]))
start = time.time()
output = m(x370, x381)
end = time.time()
print(end-start)
