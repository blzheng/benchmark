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
        self.conv2d56 = Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(36, 36), dilation=(36, 36), bias=False)
        self.batchnorm2d56 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU()

    def forward(self, x174):
        x184=self.conv2d56(x174)
        x185=self.batchnorm2d56(x184)
        x186=self.relu52(x185)
        return x186

m = M().eval()
x174 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x174)
end = time.time()
print(end-start)
