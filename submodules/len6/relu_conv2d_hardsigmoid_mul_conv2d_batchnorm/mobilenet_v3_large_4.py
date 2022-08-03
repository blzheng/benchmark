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
        self.relu15 = ReLU()
        self.conv2d44 = Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid4 = Hardsigmoid()
        self.conv2d45 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x127, x125):
        x128=self.relu15(x127)
        x129=self.conv2d44(x128)
        x130=self.hardsigmoid4(x129)
        x131=operator.mul(x130, x125)
        x132=self.conv2d45(x131)
        x133=self.batchnorm2d35(x132)
        return x133

m = M().eval()
x127 = torch.randn(torch.Size([1, 168, 1, 1]))
x125 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x127, x125)
end = time.time()
print(end-start)
