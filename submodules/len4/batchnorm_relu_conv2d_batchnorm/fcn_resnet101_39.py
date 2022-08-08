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
        self.batchnorm2d62 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)
        self.conv2d63 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x206):
        x207=self.batchnorm2d62(x206)
        x208=self.relu58(x207)
        x209=self.conv2d63(x208)
        x210=self.batchnorm2d63(x209)
        return x210

m = M().eval()
x206 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x206)
end = time.time()
print(end-start)