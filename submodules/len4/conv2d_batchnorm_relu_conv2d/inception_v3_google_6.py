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
        self.conv2d13 = Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d14 = Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)

    def forward(self, x53):
        x57=self.conv2d13(x53)
        x58=self.batchnorm2d13(x57)
        x59=torch.nn.functional.relu(x58,inplace=True)
        x60=self.conv2d14(x59)
        return x60

m = M().eval()
x53 = torch.randn(torch.Size([1, 256, 25, 25]))
start = time.time()
output = m(x53)
end = time.time()
print(end-start)
