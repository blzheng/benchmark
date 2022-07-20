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
        self.conv2d299 = Conv2d(2304, 2304, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2304, bias=False)
        self.batchnorm2d193 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x962):
        x963=self.conv2d299(x962)
        x964=self.batchnorm2d193(x963)
        return x964

m = M().eval()
x962 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x962)
end = time.time()
print(end-start)
