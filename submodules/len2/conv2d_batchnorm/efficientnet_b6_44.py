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
        self.conv2d74 = Conv2d(432, 432, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=432, bias=False)
        self.batchnorm2d44 = BatchNorm2d(432, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x231):
        x232=self.conv2d74(x231)
        x233=self.batchnorm2d44(x232)
        return x233

m = M().eval()
x231 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x231)
end = time.time()
print(end-start)
