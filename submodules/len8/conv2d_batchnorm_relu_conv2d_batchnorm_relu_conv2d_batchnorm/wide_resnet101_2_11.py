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
        self.conv2d37 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x120):
        x121=self.conv2d37(x120)
        x122=self.batchnorm2d37(x121)
        x123=self.relu34(x122)
        x124=self.conv2d38(x123)
        x125=self.batchnorm2d38(x124)
        x126=self.relu34(x125)
        x127=self.conv2d39(x126)
        x128=self.batchnorm2d39(x127)
        return x128

m = M().eval()
x120 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x120)
end = time.time()
print(end-start)
