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
        self.batchnorm2d13 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu10 = ReLU(inplace=True)

    def forward(self, x45, x48):
        x46=self.batchnorm2d13(x45)
        x49=operator.add(x46, x48)
        x50=self.relu10(x49)
        return x50

m = M().eval()
x45 = torch.randn(torch.Size([1, 512, 28, 28]))
x48 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x45, x48)
end = time.time()
print(end-start)
