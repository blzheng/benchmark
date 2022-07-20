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
        self.conv2d188 = Conv2d(1344, 1344, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=1344, bias=False)
        self.batchnorm2d112 = BatchNorm2d(1344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x592):
        x593=self.conv2d188(x592)
        x594=self.batchnorm2d112(x593)
        return x594

m = M().eval()
x592 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x592)
end = time.time()
print(end-start)
