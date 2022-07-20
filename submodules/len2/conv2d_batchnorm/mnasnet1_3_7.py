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
        self.conv2d7 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d7 = BatchNorm2d(96, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x19):
        x20=self.conv2d7(x19)
        x21=self.batchnorm2d7(x20)
        return x21

m = M().eval()
x19 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x19)
end = time.time()
print(end-start)