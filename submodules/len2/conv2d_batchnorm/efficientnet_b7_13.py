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
        self.conv2d23 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d13 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x70):
        x71=self.conv2d23(x70)
        x72=self.batchnorm2d13(x71)
        return x72

m = M().eval()
x70 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x70)
end = time.time()
print(end-start)
