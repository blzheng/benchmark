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
        self.conv2d35 = Conv2d(1232, 1232, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d23 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)

    def forward(self, x108):
        x109=self.conv2d35(x108)
        x110=self.batchnorm2d23(x109)
        x111=self.relu26(x110)
        return x111

m = M().eval()
x108 = torch.randn(torch.Size([1, 1232, 28, 28]))
start = time.time()
output = m(x108)
end = time.time()
print(end-start)
