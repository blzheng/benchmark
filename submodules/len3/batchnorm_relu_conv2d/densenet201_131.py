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
        self.batchnorm2d132 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu132 = ReLU(inplace=True)
        self.conv2d132 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x466):
        x467=self.batchnorm2d132(x466)
        x468=self.relu132(x467)
        x469=self.conv2d132(x468)
        return x469

m = M().eval()
x466 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x466)
end = time.time()
print(end-start)
