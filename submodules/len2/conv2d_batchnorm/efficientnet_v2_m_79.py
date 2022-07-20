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
        self.conv2d115 = Conv2d(1056, 1056, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1056, bias=False)
        self.batchnorm2d79 = BatchNorm2d(1056, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x371):
        x372=self.conv2d115(x371)
        x373=self.batchnorm2d79(x372)
        return x373

m = M().eval()
x371 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x371)
end = time.time()
print(end-start)
