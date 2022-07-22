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
        self.batchnorm2d95 = BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu95 = ReLU(inplace=True)
        self.conv2d95 = Conv2d(1152, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d96 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu96 = ReLU(inplace=True)
        self.conv2d96 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x337):
        x338=self.batchnorm2d95(x337)
        x339=self.relu95(x338)
        x340=self.conv2d95(x339)
        x341=self.batchnorm2d96(x340)
        x342=self.relu96(x341)
        x343=self.conv2d96(x342)
        return x343

m = M().eval()
x337 = torch.randn(torch.Size([1, 1152, 14, 14]))
start = time.time()
output = m(x337)
end = time.time()
print(end-start)
