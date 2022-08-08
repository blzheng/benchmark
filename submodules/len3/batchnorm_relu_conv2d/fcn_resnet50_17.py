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
        self.batchnorm2d29 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x96):
        x97=self.batchnorm2d29(x96)
        x98=self.relu25(x97)
        x99=self.conv2d30(x98)
        return x99

m = M().eval()
x96 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x96)
end = time.time()
print(end-start)
