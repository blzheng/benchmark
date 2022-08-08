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
        self.relu46 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d50 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x164):
        x165=self.relu46(x164)
        x166=self.conv2d50(x165)
        x167=self.batchnorm2d50(x166)
        x168=self.relu46(x167)
        x169=self.conv2d51(x168)
        x170=self.batchnorm2d51(x169)
        return x170

m = M().eval()
x164 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x164)
end = time.time()
print(end-start)
