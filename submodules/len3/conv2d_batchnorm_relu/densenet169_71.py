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
        self.conv2d144 = Conv2d(1280, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d145 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu145 = ReLU(inplace=True)

    def forward(self, x512):
        x513=self.conv2d144(x512)
        x514=self.batchnorm2d145(x513)
        x515=self.relu145(x514)
        return x515

m = M().eval()
x512 = torch.randn(torch.Size([1, 1280, 7, 7]))
start = time.time()
output = m(x512)
end = time.time()
print(end-start)
