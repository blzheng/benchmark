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
        self.batchnorm2d174 = BatchNorm2d(1504, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu174 = ReLU(inplace=True)

    def forward(self, x615):
        x616=self.batchnorm2d174(x615)
        x617=self.relu174(x616)
        return x617

m = M().eval()
x615 = torch.randn(torch.Size([1, 1504, 7, 7]))
start = time.time()
output = m(x615)
end = time.time()
print(end-start)
