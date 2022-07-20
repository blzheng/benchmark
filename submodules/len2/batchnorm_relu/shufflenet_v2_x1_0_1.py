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
        self.batchnorm2d2 = BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = ReLU(inplace=True)

    def forward(self, x7):
        x8=self.batchnorm2d2(x7)
        x9=self.relu1(x8)
        return x9

m = M().eval()
x7 = torch.randn(torch.Size([1, 58, 28, 28]))
start = time.time()
output = m(x7)
end = time.time()
print(end-start)