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
        self.batchnorm2d74 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu70 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d75 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x246):
        x247=self.batchnorm2d74(x246)
        x248=self.relu70(x247)
        x249=self.conv2d75(x248)
        x250=self.batchnorm2d75(x249)
        return x250

m = M().eval()
x246 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x246)
end = time.time()
print(end-start)
