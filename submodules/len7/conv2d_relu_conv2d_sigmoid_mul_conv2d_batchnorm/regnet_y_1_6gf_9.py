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
        self.conv2d51 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu39 = ReLU()
        self.conv2d52 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()
        self.conv2d53 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x160, x159):
        x161=self.conv2d51(x160)
        x162=self.relu39(x161)
        x163=self.conv2d52(x162)
        x164=self.sigmoid9(x163)
        x165=operator.mul(x164, x159)
        x166=self.conv2d53(x165)
        x167=self.batchnorm2d33(x166)
        return x167

m = M().eval()
x160 = torch.randn(torch.Size([1, 336, 1, 1]))
x159 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x160, x159)
end = time.time()
print(end-start)
