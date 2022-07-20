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
        self.batchnorm2d22 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu25 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(896, 896, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)

    def forward(self, x106):
        x107=self.batchnorm2d22(x106)
        x108=self.relu25(x107)
        x109=self.conv2d35(x108)
        return x109

m = M().eval()
x106 = torch.randn(torch.Size([1, 896, 28, 28]))
start = time.time()
output = m(x106)
end = time.time()
print(end-start)