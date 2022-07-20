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
        self.batchnorm2d131 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu127 = ReLU(inplace=True)
        self.conv2d132 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d132 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x434):
        x435=self.batchnorm2d131(x434)
        x436=self.relu127(x435)
        x437=self.conv2d132(x436)
        x438=self.batchnorm2d132(x437)
        return x438

m = M().eval()
x434 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x434)
end = time.time()
print(end-start)
