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
        self.batchnorm2d50 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu46 = ReLU(inplace=True)
        self.conv2d51 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        self.batchnorm2d51 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d52 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x165):
        x166=self.batchnorm2d50(x165)
        x167=self.relu46(x166)
        x168=self.conv2d51(x167)
        x169=self.batchnorm2d51(x168)
        x170=self.relu46(x169)
        x171=self.conv2d52(x170)
        return x171

m = M().eval()
x165 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x165)
end = time.time()
print(end-start)
