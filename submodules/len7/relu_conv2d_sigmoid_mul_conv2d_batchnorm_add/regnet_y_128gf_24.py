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
        self.relu99 = ReLU()
        self.conv2d127 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid24 = Sigmoid()
        self.conv2d128 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d78 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x401, x399, x393):
        x402=self.relu99(x401)
        x403=self.conv2d127(x402)
        x404=self.sigmoid24(x403)
        x405=operator.mul(x404, x399)
        x406=self.conv2d128(x405)
        x407=self.batchnorm2d78(x406)
        x408=operator.add(x393, x407)
        return x408

m = M().eval()
x401 = torch.randn(torch.Size([1, 726, 1, 1]))
x399 = torch.randn(torch.Size([1, 2904, 14, 14]))
x393 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x401, x399, x393)
end = time.time()
print(end-start)
