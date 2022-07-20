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
        self.batchnorm2d194 = BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu194 = ReLU(inplace=True)

    def forward(self, x685):
        x686=self.batchnorm2d194(x685)
        x687=self.relu194(x686)
        return x687

m = M().eval()
x685 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x685)
end = time.time()
print(end-start)
