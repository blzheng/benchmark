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
        self.conv2d184 = Conv2d(3072, 3072, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3072, bias=False)
        self.batchnorm2d110 = BatchNorm2d(3072, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x548):
        x549=self.conv2d184(x548)
        x550=self.batchnorm2d110(x549)
        return x550

m = M().eval()
x548 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x548)
end = time.time()
print(end-start)
