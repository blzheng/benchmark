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
        self.conv2d105 = Conv2d(1248, 1248, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1248, bias=False)
        self.batchnorm2d63 = BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x324):
        x325=self.conv2d105(x324)
        x326=self.batchnorm2d63(x325)
        return x326

m = M().eval()
x324 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x324)
end = time.time()
print(end-start)
