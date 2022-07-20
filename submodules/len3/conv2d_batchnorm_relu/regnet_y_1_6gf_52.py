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
        self.conv2d131 = Conv2d(888, 888, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=37, bias=False)
        self.batchnorm2d81 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu102 = ReLU(inplace=True)

    def forward(self, x414):
        x415=self.conv2d131(x414)
        x416=self.batchnorm2d81(x415)
        x417=self.relu102(x416)
        return x417

m = M().eval()
x414 = torch.randn(torch.Size([1, 888, 14, 14]))
start = time.time()
output = m(x414)
end = time.time()
print(end-start)
