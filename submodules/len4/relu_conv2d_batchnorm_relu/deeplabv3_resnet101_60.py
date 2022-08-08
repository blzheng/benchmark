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
        self.relu91 = ReLU(inplace=True)
        self.conv2d95 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d95 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x314):
        x315=self.relu91(x314)
        x316=self.conv2d95(x315)
        x317=self.batchnorm2d95(x316)
        x318=self.relu91(x317)
        return x318

m = M().eval()
x314 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x314)
end = time.time()
print(end-start)
