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
        self.sigmoid20 = Sigmoid()
        self.conv2d109 = Conv2d(1512, 1512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(1512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu84 = ReLU(inplace=True)

    def forward(self, x341, x337, x331):
        x342=self.sigmoid20(x341)
        x343=operator.mul(x342, x337)
        x344=self.conv2d109(x343)
        x345=self.batchnorm2d67(x344)
        x346=operator.add(x331, x345)
        x347=self.relu84(x346)
        return x347

m = M().eval()
x341 = torch.randn(torch.Size([1, 1512, 1, 1]))
x337 = torch.randn(torch.Size([1, 1512, 7, 7]))
x331 = torch.randn(torch.Size([1, 1512, 7, 7]))
start = time.time()
output = m(x341, x337, x331)
end = time.time()
print(end-start)
