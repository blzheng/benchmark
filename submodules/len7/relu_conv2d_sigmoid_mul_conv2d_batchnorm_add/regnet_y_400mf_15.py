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
        self.relu63 = ReLU()
        self.conv2d83 = Conv2d(110, 440, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()
        self.conv2d84 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x259, x257, x251):
        x260=self.relu63(x259)
        x261=self.conv2d83(x260)
        x262=self.sigmoid15(x261)
        x263=operator.mul(x262, x257)
        x264=self.conv2d84(x263)
        x265=self.batchnorm2d52(x264)
        x266=operator.add(x251, x265)
        return x266

m = M().eval()
x259 = torch.randn(torch.Size([1, 110, 1, 1]))
x257 = torch.randn(torch.Size([1, 440, 7, 7]))
x251 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x259, x257, x251)
end = time.time()
print(end-start)
