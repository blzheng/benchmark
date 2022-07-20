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
        self.batchnorm2d60 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU(inplace=True)
        self.conv2d61 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x195):
        x196=self.batchnorm2d60(x195)
        x197=self.relu56(x196)
        x198=self.conv2d61(x197)
        return x198

m = M().eval()
x195 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x195)
end = time.time()
print(end-start)