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
        self.relu54 = ReLU()
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.conv2d59 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d59 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu55 = ReLU()
        self.conv2d60 = Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x196):
        x197=self.relu54(x196)
        x198=self.dropout0(x197)
        x199=self.conv2d59(x198)
        x200=self.batchnorm2d59(x199)
        x201=self.relu55(x200)
        x202=self.conv2d60(x201)
        return x202

m = M().eval()
x196 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x196)
end = time.time()
print(end-start)
