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
        self.avgpool2d2 = AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x368):
        x369=self.avgpool2d2(x368)
        return x369

m = M().eval()
x368 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x368)
end = time.time()
print(end-start)
