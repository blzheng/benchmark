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
        self.relu16 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(168, 168, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=7, bias=False)
        self.batchnorm2d19 = BatchNorm2d(168, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(168, 168, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x59):
        x60=self.relu16(x59)
        x61=self.conv2d19(x60)
        x62=self.batchnorm2d19(x61)
        x63=self.relu17(x62)
        x64=self.conv2d20(x63)
        return x64

m = M().eval()
x59 = torch.randn(torch.Size([1, 168, 28, 28]))
start = time.time()
output = m(x59)
end = time.time()
print(end-start)
