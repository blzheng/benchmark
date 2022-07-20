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
        self.conv2d20 = Conv2d(256, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d21 = Conv2d(96, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x73):
        x74=self.conv2d20(x73)
        x75=self.batchnorm2d20(x74)
        x76=self.conv2d21(x75)
        return x76

m = M().eval()
x73 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x73)
end = time.time()
print(end-start)
