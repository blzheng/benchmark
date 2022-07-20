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
        self.batchnorm2d11 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu9 = ReLU(inplace=True)

    def forward(self, x34, x27):
        x35=self.batchnorm2d11(x34)
        x36=operator.add(x27, x35)
        x37=self.relu9(x36)
        return x37

m = M().eval()
x34 = torch.randn(torch.Size([1, 512, 28, 28]))
x27 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x34, x27)
end = time.time()
print(end-start)
