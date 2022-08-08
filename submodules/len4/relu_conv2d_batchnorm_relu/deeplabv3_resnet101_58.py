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
        self.relu88 = ReLU(inplace=True)
        self.conv2d92 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d92 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x304):
        x305=self.relu88(x304)
        x306=self.conv2d92(x305)
        x307=self.batchnorm2d92(x306)
        x308=self.relu88(x307)
        return x308

m = M().eval()
x304 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x304)
end = time.time()
print(end-start)
