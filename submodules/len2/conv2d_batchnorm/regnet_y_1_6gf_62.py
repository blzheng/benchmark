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
        self.conv2d100 = Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=14, bias=False)
        self.batchnorm2d62 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x316):
        x317=self.conv2d100(x316)
        x318=self.batchnorm2d62(x317)
        return x318

m = M().eval()
x316 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x316)
end = time.time()
print(end-start)
