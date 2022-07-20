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
        self.conv2d165 = Conv2d(1824, 1824, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1824, bias=False)
        self.batchnorm2d109 = BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x529):
        x530=self.conv2d165(x529)
        x531=self.batchnorm2d109(x530)
        return x531

m = M().eval()
x529 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x529)
end = time.time()
print(end-start)
