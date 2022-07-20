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
        self.conv2d134 = Conv2d(1056, 1056, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=1056, bias=False)
        self.batchnorm2d80 = BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x400):
        x401=self.conv2d134(x400)
        x402=self.batchnorm2d80(x401)
        return x402

m = M().eval()
x400 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x400)
end = time.time()
print(end-start)
