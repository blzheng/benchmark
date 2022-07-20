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
        self.relu25 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(432, 432, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=9, bias=False)
        self.batchnorm2d29 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)

    def forward(self, x91):
        x92=self.relu25(x91)
        x93=self.conv2d29(x92)
        x94=self.batchnorm2d29(x93)
        x95=self.relu26(x94)
        return x95

m = M().eval()
x91 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x91)
end = time.time()
print(end-start)
