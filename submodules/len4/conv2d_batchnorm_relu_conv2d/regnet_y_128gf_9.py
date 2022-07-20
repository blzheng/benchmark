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
        self.conv2d43 = Conv2d(1056, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu33 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(1056, 1056, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)

    def forward(self, x135):
        x136=self.conv2d43(x135)
        x137=self.batchnorm2d27(x136)
        x138=self.relu33(x137)
        x139=self.conv2d44(x138)
        return x139

m = M().eval()
x135 = torch.randn(torch.Size([1, 1056, 28, 28]))
start = time.time()
output = m(x135)
end = time.time()
print(end-start)
