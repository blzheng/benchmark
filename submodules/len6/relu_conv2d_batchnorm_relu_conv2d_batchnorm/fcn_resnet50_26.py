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
        self.relu40 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d44 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d45 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x144):
        x145=self.relu40(x144)
        x146=self.conv2d44(x145)
        x147=self.batchnorm2d44(x146)
        x148=self.relu40(x147)
        x149=self.conv2d45(x148)
        x150=self.batchnorm2d45(x149)
        return x150

m = M().eval()
x144 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x144)
end = time.time()
print(end-start)
