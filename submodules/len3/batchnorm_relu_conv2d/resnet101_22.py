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
        self.batchnorm2d37 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x121):
        x122=self.batchnorm2d37(x121)
        x123=self.relu34(x122)
        x124=self.conv2d38(x123)
        return x124

m = M().eval()
x121 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x121)
end = time.time()
print(end-start)