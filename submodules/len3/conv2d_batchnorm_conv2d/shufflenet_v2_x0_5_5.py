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
        self.conv2d15 = Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
        self.batchnorm2d15 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d16 = Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x94):
        x95=self.conv2d15(x94)
        x96=self.batchnorm2d15(x95)
        x97=self.conv2d16(x96)
        return x97

m = M().eval()
x94 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x94)
end = time.time()
print(end-start)
