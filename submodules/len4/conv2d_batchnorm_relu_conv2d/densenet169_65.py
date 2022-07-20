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
        self.conv2d134 = Conv2d(1120, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d135 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu135 = ReLU(inplace=True)
        self.conv2d135 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x477):
        x478=self.conv2d134(x477)
        x479=self.batchnorm2d135(x478)
        x480=self.relu135(x479)
        x481=self.conv2d135(x480)
        return x481

m = M().eval()
x477 = torch.randn(torch.Size([1, 1120, 7, 7]))
start = time.time()
output = m(x477)
end = time.time()
print(end-start)
