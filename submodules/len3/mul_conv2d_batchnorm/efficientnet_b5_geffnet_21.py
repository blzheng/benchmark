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
        self.conv2d107 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x313, x318):
        x319=operator.mul(x313, x318)
        x320=self.conv2d107(x319)
        x321=self.batchnorm2d63(x320)
        return x321

m = M().eval()
x313 = torch.randn(torch.Size([1, 1056, 14, 14]))
x318 = torch.randn(torch.Size([1, 1056, 1, 1]))
start = time.time()
output = m(x313, x318)
end = time.time()
print(end-start)
