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
        self.batchnorm2d19 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(208, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=26, bias=False)
        self.batchnorm2d20 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)

    def forward(self, x90):
        x91=self.batchnorm2d19(x90)
        x92=self.relu21(x91)
        x93=self.conv2d30(x92)
        x94=self.batchnorm2d20(x93)
        x95=self.relu22(x94)
        return x95

m = M().eval()
x90 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x90)
end = time.time()
print(end-start)
