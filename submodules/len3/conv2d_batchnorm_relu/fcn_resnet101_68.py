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
        self.conv2d106 = Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d105 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu101 = ReLU()

    def forward(self, x312):
        x351=self.conv2d106(x312)
        x352=self.batchnorm2d105(x351)
        x353=self.relu101(x352)
        return x353

m = M().eval()
x312 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x312)
end = time.time()
print(end-start)
