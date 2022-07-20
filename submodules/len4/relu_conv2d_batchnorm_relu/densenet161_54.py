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
        self.relu112 = ReLU(inplace=True)
        self.conv2d112 = Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d113 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu113 = ReLU(inplace=True)

    def forward(self, x399):
        x400=self.relu112(x399)
        x401=self.conv2d112(x400)
        x402=self.batchnorm2d113(x401)
        x403=self.relu113(x402)
        return x403

m = M().eval()
x399 = torch.randn(torch.Size([1, 1056, 7, 7]))
start = time.time()
output = m(x399)
end = time.time()
print(end-start)
