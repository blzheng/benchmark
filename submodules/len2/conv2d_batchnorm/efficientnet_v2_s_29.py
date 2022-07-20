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
        self.conv2d35 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
        self.batchnorm2d29 = BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x114):
        x115=self.conv2d35(x114)
        x116=self.batchnorm2d29(x115)
        return x116

m = M().eval()
x114 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x114)
end = time.time()
print(end-start)
