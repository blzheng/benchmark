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
        self.conv2d45 = Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
        self.batchnorm2d27 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x136):
        x137=self.conv2d45(x136)
        x138=self.batchnorm2d27(x137)
        return x138

m = M().eval()
x136 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x136)
end = time.time()
print(end-start)
