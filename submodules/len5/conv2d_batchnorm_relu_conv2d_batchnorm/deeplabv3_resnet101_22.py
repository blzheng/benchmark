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
        self.conv2d37 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d38 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x122):
        x123=self.conv2d37(x122)
        x124=self.batchnorm2d37(x123)
        x125=self.relu34(x124)
        x126=self.conv2d38(x125)
        x127=self.batchnorm2d38(x126)
        return x127

m = M().eval()
x122 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x122)
end = time.time()
print(end-start)
