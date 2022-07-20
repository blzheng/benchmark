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
        self.relu60 = ReLU(inplace=True)
        self.conv2d79 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)

    def forward(self, x248):
        x249=self.relu60(x248)
        x250=self.conv2d79(x249)
        x251=self.batchnorm2d49(x250)
        x252=self.relu61(x251)
        return x252

m = M().eval()
x248 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x248)
end = time.time()
print(end-start)
