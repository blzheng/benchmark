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
        self.conv2d33 = Conv2d(128, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False)
        self.batchnorm2d33 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x121):
        x122=torch.nn.functional.relu(x121,inplace=True)
        x123=self.conv2d33(x122)
        x124=self.batchnorm2d33(x123)
        x125=torch.nn.functional.relu(x124,inplace=True)
        return x125

m = M().eval()
x121 = torch.randn(torch.Size([1, 128, 12, 12]))
start = time.time()
output = m(x121)
end = time.time()
print(end-start)
