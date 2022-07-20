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
        self.conv2d117 = Conv2d(864, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d118 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x366):
        x367=self.conv2d117(x366)
        x368=self.batchnorm2d69(x367)
        x369=self.conv2d118(x368)
        return x369

m = M().eval()
x366 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x366)
end = time.time()
print(end-start)
