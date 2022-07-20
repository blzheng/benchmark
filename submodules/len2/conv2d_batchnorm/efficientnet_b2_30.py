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
        self.conv2d50 = Conv2d(528, 528, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=528, bias=False)
        self.batchnorm2d30 = BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x152):
        x153=self.conv2d50(x152)
        x154=self.batchnorm2d30(x153)
        return x154

m = M().eval()
x152 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
