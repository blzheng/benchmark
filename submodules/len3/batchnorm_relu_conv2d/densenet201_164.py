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
        self.batchnorm2d165 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu165 = ReLU(inplace=True)
        self.conv2d165 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x583):
        x584=self.batchnorm2d165(x583)
        x585=self.relu165(x584)
        x586=self.conv2d165(x585)
        return x586

m = M().eval()
x583 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x583)
end = time.time()
print(end-start)
