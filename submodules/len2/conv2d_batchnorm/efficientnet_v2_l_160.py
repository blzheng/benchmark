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
        self.conv2d244 = Conv2d(2304, 2304, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2304, bias=False)
        self.batchnorm2d160 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x786):
        x787=self.conv2d244(x786)
        x788=self.batchnorm2d160(x787)
        return x788

m = M().eval()
x786 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x786)
end = time.time()
print(end-start)
