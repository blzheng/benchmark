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
        self.conv2d23 = Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x87):
        x88=torch.nn.functional.relu(x87,inplace=True)
        x89=self.conv2d23(x88)
        x90=self.batchnorm2d23(x89)
        x91=torch.nn.functional.relu(x90,inplace=True)
        return x91

m = M().eval()
x87 = torch.randn(torch.Size([1, 64, 25, 25]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
