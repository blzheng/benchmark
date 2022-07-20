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
        self.relu101 = ReLU(inplace=True)
        self.conv2d130 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d80 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu102 = ReLU(inplace=True)

    def forward(self, x411):
        x412=self.relu101(x411)
        x413=self.conv2d130(x412)
        x414=self.batchnorm2d80(x413)
        x415=self.relu102(x414)
        return x415

m = M().eval()
x411 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x411)
end = time.time()
print(end-start)
