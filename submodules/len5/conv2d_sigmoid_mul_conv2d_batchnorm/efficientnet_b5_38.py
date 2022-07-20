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
        self.conv2d191 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid38 = Sigmoid()
        self.conv2d192 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d114 = BatchNorm2d(512, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x597, x594):
        x598=self.conv2d191(x597)
        x599=self.sigmoid38(x598)
        x600=operator.mul(x599, x594)
        x601=self.conv2d192(x600)
        x602=self.batchnorm2d114(x601)
        return x602

m = M().eval()
x597 = torch.randn(torch.Size([1, 128, 1, 1]))
x594 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x597, x594)
end = time.time()
print(end-start)
