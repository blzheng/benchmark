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
        self.relu76 = ReLU(inplace=True)
        self.conv2d99 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu77 = ReLU(inplace=True)
        self.conv2d100 = Conv2d(2904, 2904, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=11, bias=False)
        self.batchnorm2d62 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x312):
        x313=self.relu76(x312)
        x314=self.conv2d99(x313)
        x315=self.batchnorm2d61(x314)
        x316=self.relu77(x315)
        x317=self.conv2d100(x316)
        x318=self.batchnorm2d62(x317)
        return x318

m = M().eval()
x312 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x312)
end = time.time()
print(end-start)
