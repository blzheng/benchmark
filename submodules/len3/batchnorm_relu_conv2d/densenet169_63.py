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
        self.batchnorm2d64 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)
        self.conv2d64 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x228):
        x229=self.batchnorm2d64(x228)
        x230=self.relu64(x229)
        x231=self.conv2d64(x230)
        return x231

m = M().eval()
x228 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x228)
end = time.time()
print(end-start)
