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
        self.batchnorm2d44 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d45 = Conv2d(160, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
        self.batchnorm2d45 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d46 = Conv2d(160, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)
        self.batchnorm2d46 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x158):
        x159=self.batchnorm2d44(x158)
        x160=torch.nn.functional.relu(x159,inplace=True)
        x161=self.conv2d45(x160)
        x162=self.batchnorm2d45(x161)
        x163=torch.nn.functional.relu(x162,inplace=True)
        x164=self.conv2d46(x163)
        x165=self.batchnorm2d46(x164)
        x166=torch.nn.functional.relu(x165,inplace=True)
        return x166

m = M().eval()
x158 = torch.randn(torch.Size([1, 160, 12, 12]))
start = time.time()
output = m(x158)
end = time.time()
print(end-start)
