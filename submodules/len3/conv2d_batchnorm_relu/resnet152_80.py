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
        self.conv2d122 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d122 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu118 = ReLU(inplace=True)

    def forward(self, x403):
        x404=self.conv2d122(x403)
        x405=self.batchnorm2d122(x404)
        x406=self.relu118(x405)
        return x406

m = M().eval()
x403 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x403)
end = time.time()
print(end-start)
