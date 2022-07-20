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
        self.conv2d243 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d145 = BatchNorm2d(2304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x766):
        x767=self.conv2d243(x766)
        x768=self.batchnorm2d145(x767)
        return x768

m = M().eval()
x766 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x766)
end = time.time()
print(end-start)
