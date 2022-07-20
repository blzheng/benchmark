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
        self.batchnorm2d107 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu107 = ReLU(inplace=True)
        self.conv2d107 = Conv2d(1344, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x379):
        x380=self.batchnorm2d107(x379)
        x381=self.relu107(x380)
        x382=self.conv2d107(x381)
        return x382

m = M().eval()
x379 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x379)
end = time.time()
print(end-start)
