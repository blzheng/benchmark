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
        self.relu44 = ReLU(inplace=True)
        self.conv2d59 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu45 = ReLU(inplace=True)

    def forward(self, x184):
        x185=self.relu44(x184)
        x186=self.conv2d59(x185)
        x187=self.batchnorm2d37(x186)
        x188=self.relu45(x187)
        return x188

m = M().eval()
x184 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x184)
end = time.time()
print(end-start)
