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
        self.batchnorm2d20 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(336, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x73):
        x74=self.batchnorm2d20(x73)
        x75=self.relu20(x74)
        x76=self.conv2d20(x75)
        return x76

m = M().eval()
x73 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x73)
end = time.time()
print(end-start)
