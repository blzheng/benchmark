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
        self.conv2d228 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d136 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x718):
        x719=self.conv2d228(x718)
        x720=self.batchnorm2d136(x719)
        return x720

m = M().eval()
x718 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x718)
end = time.time()
print(end-start)
