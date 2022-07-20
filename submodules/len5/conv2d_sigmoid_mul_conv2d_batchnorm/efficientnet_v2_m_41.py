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
        self.conv2d232 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid41 = Sigmoid()
        self.conv2d233 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d149 = BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x741, x738):
        x742=self.conv2d232(x741)
        x743=self.sigmoid41(x742)
        x744=operator.mul(x743, x738)
        x745=self.conv2d233(x744)
        x746=self.batchnorm2d149(x745)
        return x746

m = M().eval()
x741 = torch.randn(torch.Size([1, 128, 1, 1]))
x738 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x741, x738)
end = time.time()
print(end-start)
