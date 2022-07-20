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
        self.sigmoid10 = Sigmoid()
        self.conv2d78 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(176, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x250, x246):
        x251=self.sigmoid10(x250)
        x252=operator.mul(x251, x246)
        x253=self.conv2d78(x252)
        x254=self.batchnorm2d56(x253)
        return x254

m = M().eval()
x250 = torch.randn(torch.Size([1, 1056, 1, 1]))
x246 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x250, x246)
end = time.time()
print(end-start)
