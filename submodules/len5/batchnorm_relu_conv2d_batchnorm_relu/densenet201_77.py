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
        self.batchnorm2d158 = BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu158 = ReLU(inplace=True)
        self.conv2d158 = Conv2d(1248, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d159 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu159 = ReLU(inplace=True)

    def forward(self, x559):
        x560=self.batchnorm2d158(x559)
        x561=self.relu158(x560)
        x562=self.conv2d158(x561)
        x563=self.batchnorm2d159(x562)
        x564=self.relu159(x563)
        return x564

m = M().eval()
x559 = torch.randn(torch.Size([1, 1248, 7, 7]))
start = time.time()
output = m(x559)
end = time.time()
print(end-start)
