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
        self.conv2d193 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d115 = BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x576):
        x577=self.conv2d193(x576)
        x578=self.batchnorm2d115(x577)
        return x578

m = M().eval()
x576 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x576)
end = time.time()
print(end-start)
