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
        self.conv2d51 = Conv2d(160, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(1280, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)

    def forward(self, x146):
        x147=self.conv2d51(x146)
        x148=self.batchnorm2d51(x147)
        x149=self.relu34(x148)
        return x149

m = M().eval()
x146 = torch.randn(torch.Size([1, 160, 7, 7]))
start = time.time()
output = m(x146)
end = time.time()
print(end-start)