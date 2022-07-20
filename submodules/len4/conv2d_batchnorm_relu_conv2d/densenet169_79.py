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
        self.conv2d162 = Conv2d(1568, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d163 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu163 = ReLU(inplace=True)
        self.conv2d163 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x575):
        x576=self.conv2d162(x575)
        x577=self.batchnorm2d163(x576)
        x578=self.relu163(x577)
        x579=self.conv2d163(x578)
        return x579

m = M().eval()
x575 = torch.randn(torch.Size([1, 1568, 7, 7]))
start = time.time()
output = m(x575)
end = time.time()
print(end-start)
