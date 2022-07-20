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
        self.batchnorm2d18 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu12 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=120, bias=False)
        self.batchnorm2d19 = BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)

    def forward(self, x59):
        x60=self.batchnorm2d18(x59)
        x61=self.relu12(x60)
        x62=self.conv2d19(x61)
        x63=self.batchnorm2d19(x62)
        x64=self.relu13(x63)
        return x64

m = M().eval()
x59 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x59)
end = time.time()
print(end-start)
