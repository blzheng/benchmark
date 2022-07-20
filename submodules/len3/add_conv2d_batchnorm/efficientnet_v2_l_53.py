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
        self.conv2d238 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d156 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x766, x751):
        x767=operator.add(x766, x751)
        x768=self.conv2d238(x767)
        x769=self.batchnorm2d156(x768)
        return x769

m = M().eval()
x766 = torch.randn(torch.Size([1, 384, 7, 7]))
x751 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x766, x751)
end = time.time()
print(end-start)
