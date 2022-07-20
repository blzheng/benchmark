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
        self.relu188 = ReLU(inplace=True)
        self.conv2d188 = Conv2d(1728, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d189 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x665):
        x666=self.relu188(x665)
        x667=self.conv2d188(x666)
        x668=self.batchnorm2d189(x667)
        return x668

m = M().eval()
x665 = torch.randn(torch.Size([1, 1728, 7, 7]))
start = time.time()
output = m(x665)
end = time.time()
print(end-start)
