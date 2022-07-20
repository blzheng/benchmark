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
        self.conv2d39 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d39 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu35 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x124):
        x125=self.conv2d39(x124)
        x126=self.batchnorm2d39(x125)
        x127=self.relu35(x126)
        x128=self.conv2d40(x127)
        x129=self.batchnorm2d40(x128)
        return x129

m = M().eval()
x124 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x124)
end = time.time()
print(end-start)
