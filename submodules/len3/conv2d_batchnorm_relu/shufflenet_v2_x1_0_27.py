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
        self.conv2d42 = Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)

    def forward(self, x274):
        x275=self.conv2d42(x274)
        x276=self.batchnorm2d42(x275)
        x277=self.relu27(x276)
        return x277

m = M().eval()
x274 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
