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
        self.relu63 = ReLU(inplace=True)
        self.conv2d67 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)
        self.conv2d68 = Conv2d(1344, 1344, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)
        self.batchnorm2d68 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x218):
        x219=self.relu63(x218)
        x220=self.conv2d67(x219)
        x221=self.batchnorm2d67(x220)
        x222=self.relu64(x221)
        x223=self.conv2d68(x222)
        x224=self.batchnorm2d68(x223)
        x225=self.relu65(x224)
        x226=self.conv2d69(x225)
        x227=self.batchnorm2d69(x226)
        return x227

m = M().eval()
x218 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x218)
end = time.time()
print(end-start)
