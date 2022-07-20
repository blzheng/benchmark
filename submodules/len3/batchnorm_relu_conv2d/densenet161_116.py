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
        self.batchnorm2d117 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu117 = ReLU(inplace=True)
        self.conv2d117 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x415):
        x416=self.batchnorm2d117(x415)
        x417=self.relu117(x416)
        x418=self.conv2d117(x417)
        return x418

m = M().eval()
x415 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x415)
end = time.time()
print(end-start)
