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
        self.conv2d128 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d76 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x402):
        x403=self.conv2d128(x402)
        x404=self.batchnorm2d76(x403)
        return x404

m = M().eval()
x402 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x402)
end = time.time()
print(end-start)
