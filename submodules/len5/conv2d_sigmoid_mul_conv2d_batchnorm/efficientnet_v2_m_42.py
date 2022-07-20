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
        self.conv2d237 = Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid42 = Sigmoid()
        self.conv2d238 = Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d152 = BatchNorm2d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x757, x754):
        x758=self.conv2d237(x757)
        x759=self.sigmoid42(x758)
        x760=operator.mul(x759, x754)
        x761=self.conv2d238(x760)
        x762=self.batchnorm2d152(x761)
        return x762

m = M().eval()
x757 = torch.randn(torch.Size([1, 128, 1, 1]))
x754 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x757, x754)
end = time.time()
print(end-start)
