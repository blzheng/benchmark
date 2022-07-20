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
        self.conv2d66 = Conv2d(48, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d67 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x220, x217):
        x221=self.conv2d66(x220)
        x222=self.sigmoid6(x221)
        x223=operator.mul(x222, x217)
        x224=self.conv2d67(x223)
        x225=self.batchnorm2d53(x224)
        return x225

m = M().eval()
x220 = torch.randn(torch.Size([1, 48, 1, 1]))
x217 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x220, x217)
end = time.time()
print(end-start)
