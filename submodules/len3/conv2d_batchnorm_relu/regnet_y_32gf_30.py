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
        self.conv2d75 = Conv2d(1392, 1392, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=6, bias=False)
        self.batchnorm2d47 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)

    def forward(self, x236):
        x237=self.conv2d75(x236)
        x238=self.batchnorm2d47(x237)
        x239=self.relu58(x238)
        return x239

m = M().eval()
x236 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x236)
end = time.time()
print(end-start)
