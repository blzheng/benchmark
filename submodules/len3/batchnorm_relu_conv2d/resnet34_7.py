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
        self.batchnorm2d16 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x56):
        x57=self.batchnorm2d16(x56)
        x58=self.relu15(x57)
        x59=self.conv2d17(x58)
        return x59

m = M().eval()
x56 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x56)
end = time.time()
print(end-start)
