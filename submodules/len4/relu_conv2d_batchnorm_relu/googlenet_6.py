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
        self.conv2d19 = Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x76):
        x77=torch.nn.functional.relu(x76,inplace=True)
        x78=self.conv2d19(x77)
        x79=self.batchnorm2d19(x78)
        x80=torch.nn.functional.relu(x79,inplace=True)
        return x80

m = M().eval()
x76 = torch.randn(torch.Size([1, 16, 14, 14]))
start = time.time()
output = m(x76)
end = time.time()
print(end-start)
