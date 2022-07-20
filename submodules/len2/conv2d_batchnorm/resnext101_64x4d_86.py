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
        self.conv2d86 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.batchnorm2d86 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x283):
        x284=self.conv2d86(x283)
        x285=self.batchnorm2d86(x284)
        return x285

m = M().eval()
x283 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x283)
end = time.time()
print(end-start)
