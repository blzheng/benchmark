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
        self.conv2d52 = Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
        self.batchnorm2d40 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x151):
        x152=self.conv2d52(x151)
        x153=self.batchnorm2d40(x152)
        return x153

m = M().eval()
x151 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x151)
end = time.time()
print(end-start)
