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

    def forward(self, x30):
        x49=torch._C._nn.avg_pool2d(x30,kernel_size=3, stride=1, padding=1)
        return x49

m = M().eval()
x30 = torch.randn(torch.Size([1, 192, 25, 25]))
start = time.time()
output = m(x30)
end = time.time()
print(end-start)
