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
        self.batchnorm2d67 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)
        self.conv2d68 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        self.batchnorm2d68 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x223):
        x224=self.batchnorm2d67(x223)
        x225=self.relu64(x224)
        x226=self.conv2d68(x225)
        x227=self.batchnorm2d68(x226)
        x228=self.relu64(x227)
        return x228

m = M().eval()
x223 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x223)
end = time.time()
print(end-start)
