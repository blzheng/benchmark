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
        self.batchnorm2d28 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d29 = Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x109):
        x110=self.batchnorm2d28(x109)
        x111=torch.nn.functional.relu(x110,inplace=True)
        x112=self.conv2d29(x111)
        return x112

m = M().eval()
x109 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
