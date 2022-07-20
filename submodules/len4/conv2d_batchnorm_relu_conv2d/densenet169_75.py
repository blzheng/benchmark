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
        self.conv2d154 = Conv2d(1440, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d155 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu155 = ReLU(inplace=True)
        self.conv2d155 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x547):
        x548=self.conv2d154(x547)
        x549=self.batchnorm2d155(x548)
        x550=self.relu155(x549)
        x551=self.conv2d155(x550)
        return x551

m = M().eval()
x547 = torch.randn(torch.Size([1, 1440, 7, 7]))
start = time.time()
output = m(x547)
end = time.time()
print(end-start)
