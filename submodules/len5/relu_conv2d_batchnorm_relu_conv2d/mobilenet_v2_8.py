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
        self.relu616 = ReLU6(inplace=True)
        self.conv2d25 = Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.batchnorm2d25 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu617 = ReLU6(inplace=True)
        self.conv2d26 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x70):
        x71=self.relu616(x70)
        x72=self.conv2d25(x71)
        x73=self.batchnorm2d25(x72)
        x74=self.relu617(x73)
        x75=self.conv2d26(x74)
        return x75

m = M().eval()
x70 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x70)
end = time.time()
print(end-start)