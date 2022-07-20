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
        self.relu13 = ReLU(inplace=True)
        self.conv2d19 = Conv2d(216, 216, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=9, bias=False)
        self.batchnorm2d13 = BatchNorm2d(216, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)

    def forward(self, x57):
        x58=self.relu13(x57)
        x59=self.conv2d19(x58)
        x60=self.batchnorm2d13(x59)
        x61=self.relu14(x60)
        return x61

m = M().eval()
x57 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x57)
end = time.time()
print(end-start)
