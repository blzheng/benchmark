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
        self.batchnorm2d147 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu147 = ReLU(inplace=True)
        self.conv2d147 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x520):
        x521=self.batchnorm2d147(x520)
        x522=self.relu147(x521)
        x523=self.conv2d147(x522)
        return x523

m = M().eval()
x520 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x520)
end = time.time()
print(end-start)
