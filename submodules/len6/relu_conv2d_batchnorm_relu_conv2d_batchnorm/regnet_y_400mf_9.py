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
        self.relu52 = ReLU(inplace=True)
        self.conv2d70 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(440, 440, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=55, bias=False)
        self.batchnorm2d45 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x218):
        x219=self.relu52(x218)
        x220=self.conv2d70(x219)
        x221=self.batchnorm2d44(x220)
        x222=self.relu53(x221)
        x223=self.conv2d71(x222)
        x224=self.batchnorm2d45(x223)
        return x224

m = M().eval()
x218 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x218)
end = time.time()
print(end-start)
