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
        self.conv2d58 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x174):
        x175=self.conv2d58(x174)
        x176=self.batchnorm2d34(x175)
        return x176

m = M().eval()
x174 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
