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
        self.relu26 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x95):
        x96=self.relu26(x95)
        x97=self.conv2d26(x96)
        x98=self.batchnorm2d27(x97)
        return x98

m = M().eval()
x95 = torch.randn(torch.Size([1, 320, 28, 28]))
start = time.time()
output = m(x95)
end = time.time()
print(end-start)
