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
        self.conv2d39 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d23 = BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x121):
        x122=self.conv2d39(x121)
        x123=self.batchnorm2d23(x122)
        return x123

m = M().eval()
x121 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x121)
end = time.time()
print(end-start)
