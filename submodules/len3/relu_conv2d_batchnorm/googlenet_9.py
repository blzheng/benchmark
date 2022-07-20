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
        self.conv2d29 = Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x110):
        x111=torch.nn.functional.relu(x110,inplace=True)
        x112=self.conv2d29(x111)
        x113=self.batchnorm2d29(x112)
        return x113

m = M().eval()
x110 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x110)
end = time.time()
print(end-start)
