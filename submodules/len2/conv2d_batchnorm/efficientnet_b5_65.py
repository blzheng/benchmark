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
        self.conv2d109 = Conv2d(1056, 1056, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1056, bias=False)
        self.batchnorm2d65 = BatchNorm2d(1056, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x339):
        x340=self.conv2d109(x339)
        x341=self.batchnorm2d65(x340)
        return x341

m = M().eval()
x339 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x339)
end = time.time()
print(end-start)
