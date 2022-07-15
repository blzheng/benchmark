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
        self.avgpool2d1 = AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x139):
        x140=self.avgpool2d1(x139)
        return x140

m = M().eval()
x139 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x139)
end = time.time()
print(end-start)
