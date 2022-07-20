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
        self.batchnorm2d109 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu109 = ReLU(inplace=True)
        self.conv2d109 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x387):
        x388=self.batchnorm2d109(x387)
        x389=self.relu109(x388)
        x390=self.conv2d109(x389)
        return x390

m = M().eval()
x387 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x387)
end = time.time()
print(end-start)
