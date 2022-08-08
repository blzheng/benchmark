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

    def forward(self, x306, x294):
        x307=torch.matmul(x306, x294)
        return x307

m = M().eval()
x306 = torch.randn(torch.Size([1, 4, 384, 384]))
x294 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x306, x294)
end = time.time()
print(end-start)
