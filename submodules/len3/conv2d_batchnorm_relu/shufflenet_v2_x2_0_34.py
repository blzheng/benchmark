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
        self.conv2d52 = Conv2d(488, 488, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)

    def forward(self, x343):
        x344=self.conv2d52(x343)
        x345=self.batchnorm2d52(x344)
        x346=self.relu34(x345)
        return x346

m = M().eval()
x343 = torch.randn(torch.Size([1, 488, 7, 7]))
start = time.time()
output = m(x343)
end = time.time()
print(end-start)
