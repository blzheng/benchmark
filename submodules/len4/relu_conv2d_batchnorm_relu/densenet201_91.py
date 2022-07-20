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
        self.relu186 = ReLU(inplace=True)
        self.conv2d186 = Conv2d(1696, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d187 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu187 = ReLU(inplace=True)

    def forward(self, x658):
        x659=self.relu186(x658)
        x660=self.conv2d186(x659)
        x661=self.batchnorm2d187(x660)
        x662=self.relu187(x661)
        return x662

m = M().eval()
x658 = torch.randn(torch.Size([1, 1696, 7, 7]))
start = time.time()
output = m(x658)
end = time.time()
print(end-start)
