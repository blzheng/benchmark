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
        self.conv2d26 = Conv2d(408, 408, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=17, bias=False)
        self.batchnorm2d26 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x82):
        x83=self.conv2d26(x82)
        x84=self.batchnorm2d26(x83)
        return x84

m = M().eval()
x82 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x82)
end = time.time()
print(end-start)
