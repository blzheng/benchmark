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
        self.relu25 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x97):
        x98=self.relu25(x97)
        x99=self.conv2d30(x98)
        x100=self.batchnorm2d30(x99)
        return x100

m = M().eval()
x97 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x97)
end = time.time()
print(end-start)
