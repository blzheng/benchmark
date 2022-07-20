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
        self.relu21 = ReLU(inplace=True)
        self.conv2d29 = Conv2d(696, 696, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3, bias=False)
        self.batchnorm2d19 = BatchNorm2d(696, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu22 = ReLU(inplace=True)

    def forward(self, x89):
        x90=self.relu21(x89)
        x91=self.conv2d29(x90)
        x92=self.batchnorm2d19(x91)
        x93=self.relu22(x92)
        return x93

m = M().eval()
x89 = torch.randn(torch.Size([1, 696, 28, 28]))
start = time.time()
output = m(x89)
end = time.time()
print(end-start)
