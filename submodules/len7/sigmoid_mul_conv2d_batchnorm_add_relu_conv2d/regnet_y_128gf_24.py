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
        self.sigmoid24 = Sigmoid()
        self.conv2d128 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d78 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu100 = ReLU(inplace=True)
        self.conv2d129 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x403, x399, x393):
        x404=self.sigmoid24(x403)
        x405=operator.mul(x404, x399)
        x406=self.conv2d128(x405)
        x407=self.batchnorm2d78(x406)
        x408=operator.add(x393, x407)
        x409=self.relu100(x408)
        x410=self.conv2d129(x409)
        return x410

m = M().eval()
x403 = torch.randn(torch.Size([1, 2904, 1, 1]))
x399 = torch.randn(torch.Size([1, 2904, 14, 14]))
x393 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x403, x399, x393)
end = time.time()
print(end-start)
