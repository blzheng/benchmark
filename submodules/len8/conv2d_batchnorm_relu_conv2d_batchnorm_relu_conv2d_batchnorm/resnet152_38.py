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
        self.conv2d118 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d118 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu115 = ReLU(inplace=True)
        self.conv2d119 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d119 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d120 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d120 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x390):
        x391=self.conv2d118(x390)
        x392=self.batchnorm2d118(x391)
        x393=self.relu115(x392)
        x394=self.conv2d119(x393)
        x395=self.batchnorm2d119(x394)
        x396=self.relu115(x395)
        x397=self.conv2d120(x396)
        x398=self.batchnorm2d120(x397)
        return x398

m = M().eval()
x390 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x390)
end = time.time()
print(end-start)
