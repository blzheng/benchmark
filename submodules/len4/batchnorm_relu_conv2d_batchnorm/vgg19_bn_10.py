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
        self.batchnorm2d14 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d15 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x47):
        x48=self.batchnorm2d14(x47)
        x49=self.relu14(x48)
        x50=self.conv2d15(x49)
        x51=self.batchnorm2d15(x50)
        return x51

m = M().eval()
x47 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x47)
end = time.time()
print(end-start)