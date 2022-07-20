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
        self.conv2d20 = Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d20 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x64):
        x65=self.conv2d20(x64)
        x66=self.batchnorm2d20(x65)
        return x66

m = M().eval()
x64 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x64)
end = time.time()
print(end-start)
