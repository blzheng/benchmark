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
        self.batchnorm2d122 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu118 = ReLU(inplace=True)
        self.conv2d123 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x404):
        x405=self.batchnorm2d122(x404)
        x406=self.relu118(x405)
        x407=self.conv2d123(x406)
        return x407

m = M().eval()
x404 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x404)
end = time.time()
print(end-start)
