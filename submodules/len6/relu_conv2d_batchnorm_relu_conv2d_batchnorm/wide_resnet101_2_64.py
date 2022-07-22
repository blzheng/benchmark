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
        self.relu97 = ReLU(inplace=True)
        self.conv2d102 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d102 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d103 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d103 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x334):
        x335=self.relu97(x334)
        x336=self.conv2d102(x335)
        x337=self.batchnorm2d102(x336)
        x338=self.relu97(x337)
        x339=self.conv2d103(x338)
        x340=self.batchnorm2d103(x339)
        return x340

m = M().eval()
x334 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x334)
end = time.time()
print(end-start)
