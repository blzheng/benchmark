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
        self.conv2d87 = Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.batchnorm2d87 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x298):
        x299=self.conv2d87(x298)
        x300=self.batchnorm2d87(x299)
        x301=torch.nn.functional.relu(x300,inplace=True)
        return x301

m = M().eval()
x298 = torch.randn(torch.Size([1, 384, 5, 5]))
start = time.time()
output = m(x298)
end = time.time()
print(end-start)
