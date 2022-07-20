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
        self.batchnorm2d13 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x44):
        x45=self.batchnorm2d13(x44)
        x46=self.relu13(x45)
        x47=self.conv2d14(x46)
        return x47

m = M().eval()
x44 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x44)
end = time.time()
print(end-start)
