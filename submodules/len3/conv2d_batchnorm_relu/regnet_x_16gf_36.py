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
        self.conv2d56 = Conv2d(896, 896, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=7, bias=False)
        self.batchnorm2d56 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu53 = ReLU(inplace=True)

    def forward(self, x182):
        x183=self.conv2d56(x182)
        x184=self.batchnorm2d56(x183)
        x185=self.relu53(x184)
        return x185

m = M().eval()
x182 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)
