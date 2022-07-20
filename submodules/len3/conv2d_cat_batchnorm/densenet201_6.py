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
        self.conv2d17 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x64, x51, x58, x72):
        x65=self.conv2d17(x64)
        x73=torch.cat([x51, x58, x65, x72], 1)
        x74=self.batchnorm2d20(x73)
        return x74

m = M().eval()
x64 = torch.randn(torch.Size([1, 128, 28, 28]))
x51 = torch.randn(torch.Size([1, 128, 28, 28]))
x58 = torch.randn(torch.Size([1, 32, 28, 28]))
x72 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x64, x51, x58, x72)
end = time.time()
print(end-start)
