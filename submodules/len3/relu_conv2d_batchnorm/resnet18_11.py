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
        self.relu11 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x49):
        x50=self.relu11(x49)
        x51=self.conv2d15(x50)
        x52=self.batchnorm2d15(x51)
        return x52

m = M().eval()
x49 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x49)
end = time.time()
print(end-start)
