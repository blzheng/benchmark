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
        self.conv2d27 = Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
        self.batchnorm2d19 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x78):
        x79=self.conv2d27(x78)
        x80=self.batchnorm2d19(x79)
        return x80

m = M().eval()
x78 = torch.randn(torch.Size([1, 120, 14, 14]))
start = time.time()
output = m(x78)
end = time.time()
print(end-start)
