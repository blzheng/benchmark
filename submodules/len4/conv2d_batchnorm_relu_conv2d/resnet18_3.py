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
        self.conv2d8 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x27):
        x28=self.conv2d8(x27)
        x29=self.batchnorm2d8(x28)
        x30=self.relu7(x29)
        x31=self.conv2d9(x30)
        return x31

m = M().eval()
x27 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x27)
end = time.time()
print(end-start)
