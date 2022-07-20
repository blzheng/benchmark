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
        self.relu110 = ReLU(inplace=True)
        self.conv2d110 = Conv2d(864, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d111 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x392):
        x393=self.relu110(x392)
        x394=self.conv2d110(x393)
        x395=self.batchnorm2d111(x394)
        return x395

m = M().eval()
x392 = torch.randn(torch.Size([1, 864, 7, 7]))
start = time.time()
output = m(x392)
end = time.time()
print(end-start)
