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
        self.conv2d214 = Conv2d(3456, 3456, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3456, bias=False)
        self.batchnorm2d128 = BatchNorm2d(3456, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x638):
        x639=self.conv2d214(x638)
        x640=self.batchnorm2d128(x639)
        return x640

m = M().eval()
x638 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x638)
end = time.time()
print(end-start)
