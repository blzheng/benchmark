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
        self.conv2d25 = Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d19 = BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x75):
        x76=self.conv2d25(x75)
        x77=self.batchnorm2d19(x76)
        return x77

m = M().eval()
x75 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x75)
end = time.time()
print(end-start)
