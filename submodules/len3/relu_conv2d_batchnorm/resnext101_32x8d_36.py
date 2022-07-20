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
        self.relu37 = ReLU(inplace=True)
        self.conv2d41 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d41 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x132):
        x133=self.relu37(x132)
        x134=self.conv2d41(x133)
        x135=self.batchnorm2d41(x134)
        return x135

m = M().eval()
x132 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x132)
end = time.time()
print(end-start)
