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
        self.batchnorm2d15 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4, bias=False)

    def forward(self, x48):
        x49=self.batchnorm2d15(x48)
        x50=self.relu13(x49)
        x51=self.conv2d16(x50)
        return x51

m = M().eval()
x48 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x48)
end = time.time()
print(end-start)
