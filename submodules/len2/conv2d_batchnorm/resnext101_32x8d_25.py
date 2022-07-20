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
        self.conv2d25 = Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d25 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x81):
        x82=self.conv2d25(x81)
        x83=self.batchnorm2d25(x82)
        return x83

m = M().eval()
x81 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x81)
end = time.time()
print(end-start)
