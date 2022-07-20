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
        self.batchnorm2d46 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu57 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d47 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x234):
        x235=self.batchnorm2d46(x234)
        x236=self.relu57(x235)
        x237=self.conv2d75(x236)
        x238=self.batchnorm2d47(x237)
        return x238

m = M().eval()
x234 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x234)
end = time.time()
print(end-start)
