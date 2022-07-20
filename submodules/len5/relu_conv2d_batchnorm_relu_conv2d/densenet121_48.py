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
        self.relu100 = ReLU(inplace=True)
        self.conv2d100 = Conv2d(704, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d101 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu101 = ReLU(inplace=True)
        self.conv2d101 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x357):
        x358=self.relu100(x357)
        x359=self.conv2d100(x358)
        x360=self.batchnorm2d101(x359)
        x361=self.relu101(x360)
        x362=self.conv2d101(x361)
        return x362

m = M().eval()
x357 = torch.randn(torch.Size([1, 704, 7, 7]))
start = time.time()
output = m(x357)
end = time.time()
print(end-start)
