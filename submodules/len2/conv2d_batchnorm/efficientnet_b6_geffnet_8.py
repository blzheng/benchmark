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
        self.conv2d14 = Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d8 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x44):
        x45=self.conv2d14(x44)
        x46=self.batchnorm2d8(x45)
        return x46

m = M().eval()
x44 = torch.randn(torch.Size([1, 192, 112, 112]))
start = time.time()
output = m(x44)
end = time.time()
print(end-start)
