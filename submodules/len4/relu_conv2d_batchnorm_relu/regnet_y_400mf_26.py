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
        self.relu60 = ReLU(inplace=True)
        self.conv2d80 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU(inplace=True)

    def forward(self, x250):
        x251=self.relu60(x250)
        x252=self.conv2d80(x251)
        x253=self.batchnorm2d50(x252)
        x254=self.relu61(x253)
        return x254

m = M().eval()
x250 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x250)
end = time.time()
print(end-start)