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
        self.batchnorm2d52 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d53 = Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x190):
        x191=self.batchnorm2d52(x190)
        x192=torch.nn.functional.relu(x191,inplace=True)
        x193=self.conv2d53(x192)
        return x193

m = M().eval()
x190 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x190)
end = time.time()
print(end-start)
