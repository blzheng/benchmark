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
        self.batchnorm2d20 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d21 = Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batchnorm2d21 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x80):
        x81=self.batchnorm2d20(x80)
        x82=torch.nn.functional.relu(x81,inplace=True)
        x83=self.conv2d21(x82)
        x84=self.batchnorm2d21(x83)
        x85=torch.nn.functional.relu(x84,inplace=True)
        return x85

m = M().eval()
x80 = torch.randn(torch.Size([1, 48, 25, 25]))
start = time.time()
output = m(x80)
end = time.time()
print(end-start)
