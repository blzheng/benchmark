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
        self.conv2d198 = Conv2d(2304, 2304, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2304, bias=False)
        self.batchnorm2d118 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x592):
        x593=self.conv2d198(x592)
        x594=self.batchnorm2d118(x593)
        return x594

m = M().eval()
x592 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x592)
end = time.time()
print(end-start)
