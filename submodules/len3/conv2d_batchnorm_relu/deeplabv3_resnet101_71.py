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
        self.conv2d108 = Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d108 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu104 = ReLU()

    def forward(self, x359):
        x360=self.conv2d108(x359)
        x361=self.batchnorm2d108(x360)
        x362=self.relu104(x361)
        return x362

m = M().eval()
x359 = torch.randn(torch.Size([1, 2048, 1, 1]))
start = time.time()
output = m(x359)
end = time.time()
print(end-start)
