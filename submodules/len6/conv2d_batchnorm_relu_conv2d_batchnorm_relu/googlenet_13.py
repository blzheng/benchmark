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
        self.conv2d40 = Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d41 = Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x145):
        x149=self.conv2d40(x145)
        x150=self.batchnorm2d40(x149)
        x151=torch.nn.functional.relu(x150,inplace=True)
        x152=self.conv2d41(x151)
        x153=self.batchnorm2d41(x152)
        x154=torch.nn.functional.relu(x153,inplace=True)
        return x154

m = M().eval()
x145 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x145)
end = time.time()
print(end-start)
