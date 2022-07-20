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
        self.batchnorm2d68 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(720, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x223):
        x224=self.batchnorm2d68(x223)
        x225=self.relu65(x224)
        x226=self.conv2d69(x225)
        x227=self.batchnorm2d69(x226)
        return x227

m = M().eval()
x223 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x223)
end = time.time()
print(end-start)
