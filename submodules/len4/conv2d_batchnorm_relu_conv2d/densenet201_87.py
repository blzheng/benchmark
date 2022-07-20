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
        self.conv2d178 = Conv2d(1568, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d179 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu179 = ReLU(inplace=True)
        self.conv2d179 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x631):
        x632=self.conv2d178(x631)
        x633=self.batchnorm2d179(x632)
        x634=self.relu179(x633)
        x635=self.conv2d179(x634)
        return x635

m = M().eval()
x631 = torch.randn(torch.Size([1, 1568, 7, 7]))
start = time.time()
output = m(x631)
end = time.time()
print(end-start)
