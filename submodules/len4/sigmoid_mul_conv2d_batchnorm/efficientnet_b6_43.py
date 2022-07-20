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
        self.sigmoid43 = Sigmoid()
        self.conv2d217 = Conv2d(3456, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d129 = BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x678, x674):
        x679=self.sigmoid43(x678)
        x680=operator.mul(x679, x674)
        x681=self.conv2d217(x680)
        x682=self.batchnorm2d129(x681)
        return x682

m = M().eval()
x678 = torch.randn(torch.Size([1, 3456, 1, 1]))
x674 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x678, x674)
end = time.time()
print(end-start)
