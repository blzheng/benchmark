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
        self.relu10 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x46, x48):
        x49=operator.add(x46, x48)
        x50=self.relu10(x49)
        x51=self.conv2d15(x50)
        x52=self.batchnorm2d15(x51)
        return x52

m = M().eval()
x46 = torch.randn(torch.Size([1, 512, 28, 28]))
x48 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x46, x48)
end = time.time()
print(end-start)
