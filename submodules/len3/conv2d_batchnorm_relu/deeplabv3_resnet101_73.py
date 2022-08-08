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
        self.conv2d110 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d110 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu106 = ReLU()

    def forward(self, x368):
        x369=self.conv2d110(x368)
        x370=self.batchnorm2d110(x369)
        x371=self.relu106(x370)
        return x371

m = M().eval()
x368 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x368)
end = time.time()
print(end-start)
