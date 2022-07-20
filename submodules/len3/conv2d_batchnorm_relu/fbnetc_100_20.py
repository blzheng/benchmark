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
        self.conv2d29 = Conv2d(192, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d29 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)

    def forward(self, x93):
        x94=self.conv2d29(x93)
        x95=self.batchnorm2d29(x94)
        x96=self.relu20(x95)
        return x96

m = M().eval()
x93 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x93)
end = time.time()
print(end-start)
