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
        self.batchnorm2d54 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU(inplace=True)
        self.conv2d89 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x278, x265):
        x279=self.batchnorm2d54(x278)
        x280=operator.add(x265, x279)
        x281=self.relu68(x280)
        x282=self.conv2d89(x281)
        return x282

m = M().eval()
x278 = torch.randn(torch.Size([1, 2904, 14, 14]))
x265 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x278, x265)
end = time.time()
print(end-start)
