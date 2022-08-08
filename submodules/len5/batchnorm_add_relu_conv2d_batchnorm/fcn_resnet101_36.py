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
        self.batchnorm2d103 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu97 = ReLU(inplace=True)
        self.conv2d104 = Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d104 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x341, x334):
        x342=self.batchnorm2d103(x341)
        x343=operator.add(x342, x334)
        x344=self.relu97(x343)
        x345=self.conv2d104(x344)
        x346=self.batchnorm2d104(x345)
        return x346

m = M().eval()
x341 = torch.randn(torch.Size([1, 2048, 28, 28]))
x334 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x341, x334)
end = time.time()
print(end-start)
