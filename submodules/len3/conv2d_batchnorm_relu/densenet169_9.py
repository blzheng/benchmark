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
        self.conv2d18 = Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)

    def forward(self, x68):
        x69=self.conv2d18(x68)
        x70=self.batchnorm2d19(x69)
        x71=self.relu19(x70)
        return x71

m = M().eval()
x68 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x68)
end = time.time()
print(end-start)
