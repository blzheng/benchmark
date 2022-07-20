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
        self.conv2d45 = Conv2d(336, 336, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=14, bias=False)
        self.batchnorm2d29 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x140):
        x141=self.conv2d45(x140)
        x142=self.batchnorm2d29(x141)
        return x142

m = M().eval()
x140 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x140)
end = time.time()
print(end-start)
