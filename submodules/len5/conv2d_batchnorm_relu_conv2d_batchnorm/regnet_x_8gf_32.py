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
        self.conv2d50 = Conv2d(720, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d50 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu47 = ReLU(inplace=True)
        self.conv2d51 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x162):
        x163=self.conv2d50(x162)
        x164=self.batchnorm2d50(x163)
        x165=self.relu47(x164)
        x166=self.conv2d51(x165)
        x167=self.batchnorm2d51(x166)
        return x167

m = M().eval()
x162 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x162)
end = time.time()
print(end-start)
