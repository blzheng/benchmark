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
        self.batchnorm2d32 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu29 = ReLU(inplace=True)
        self.conv2d33 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x109):
        x110=self.batchnorm2d32(x109)
        x111=self.relu29(x110)
        x112=self.conv2d33(x111)
        x113=self.batchnorm2d33(x112)
        return x113

m = M().eval()
x109 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
