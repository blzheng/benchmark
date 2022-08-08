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
        self.batchnorm2d19 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu16 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x64):
        x65=self.batchnorm2d19(x64)
        x66=self.relu16(x65)
        x67=self.conv2d20(x66)
        return x67

m = M().eval()
x64 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x64)
end = time.time()
print(end-start)
