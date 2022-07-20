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
        self.conv2d102 = Conv2d(768, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d60 = BatchNorm2d(176, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d103 = Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x318):
        x319=self.conv2d102(x318)
        x320=self.batchnorm2d60(x319)
        x321=self.conv2d103(x320)
        return x321

m = M().eval()
x318 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x318)
end = time.time()
print(end-start)
