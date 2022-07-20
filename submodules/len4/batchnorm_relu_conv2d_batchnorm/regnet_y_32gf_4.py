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
        self.batchnorm2d12 = BatchNorm2d(696, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(696, 696, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3, bias=False)
        self.batchnorm2d13 = BatchNorm2d(696, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x56):
        x57=self.batchnorm2d12(x56)
        x58=self.relu13(x57)
        x59=self.conv2d19(x58)
        x60=self.batchnorm2d13(x59)
        return x60

m = M().eval()
x56 = torch.randn(torch.Size([1, 696, 28, 28]))
start = time.time()
output = m(x56)
end = time.time()
print(end-start)
