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
        self.conv2d121 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d121 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu118 = ReLU(inplace=True)

    def forward(self, x400):
        x401=self.conv2d121(x400)
        x402=self.batchnorm2d121(x401)
        x403=self.relu118(x402)
        return x403

m = M().eval()
x400 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x400)
end = time.time()
print(end-start)
