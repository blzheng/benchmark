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
        self.batchnorm2d18 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(72, 72, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=72, bias=False)
        self.batchnorm2d19 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)

    def forward(self, x52):
        x53=self.batchnorm2d18(x52)
        x54=self.relu12(x53)
        x55=self.conv2d19(x54)
        x56=self.batchnorm2d19(x55)
        x57=self.relu13(x56)
        return x57

m = M().eval()
x52 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x52)
end = time.time()
print(end-start)