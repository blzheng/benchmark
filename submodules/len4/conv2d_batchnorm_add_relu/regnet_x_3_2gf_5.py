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
        self.conv2d14 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)

    def forward(self, x43, x37):
        x44=self.conv2d14(x43)
        x45=self.batchnorm2d14(x44)
        x46=operator.add(x37, x45)
        x47=self.relu12(x46)
        return x47

m = M().eval()
x43 = torch.randn(torch.Size([1, 192, 28, 28]))
x37 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x43, x37)
end = time.time()
print(end-start)
