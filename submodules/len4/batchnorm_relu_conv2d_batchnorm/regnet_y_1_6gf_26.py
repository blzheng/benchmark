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
        self.batchnorm2d80 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu101 = ReLU(inplace=True)
        self.conv2d131 = Conv2d(888, 888, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=37, bias=False)
        self.batchnorm2d81 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x412):
        x413=self.batchnorm2d80(x412)
        x414=self.relu101(x413)
        x415=self.conv2d131(x414)
        x416=self.batchnorm2d81(x415)
        return x416

m = M().eval()
x412 = torch.randn(torch.Size([1, 888, 14, 14]))
start = time.time()
output = m(x412)
end = time.time()
print(end-start)
