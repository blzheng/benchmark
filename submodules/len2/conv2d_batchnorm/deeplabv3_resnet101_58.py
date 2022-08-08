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
        self.conv2d58 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d58 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x192):
        x193=self.conv2d58(x192)
        x194=self.batchnorm2d58(x193)
        return x194

m = M().eval()
x192 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x192)
end = time.time()
print(end-start)
