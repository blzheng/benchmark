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
        self.conv2d103 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d103 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu100 = ReLU(inplace=True)
        self.conv2d104 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d104 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d105 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d105 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x340):
        x341=self.conv2d103(x340)
        x342=self.batchnorm2d103(x341)
        x343=self.relu100(x342)
        x344=self.conv2d104(x343)
        x345=self.batchnorm2d104(x344)
        x346=self.relu100(x345)
        x347=self.conv2d105(x346)
        x348=self.batchnorm2d105(x347)
        return x348

m = M().eval()
x340 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x340)
end = time.time()
print(end-start)
