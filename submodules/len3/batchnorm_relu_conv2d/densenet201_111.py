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
        self.batchnorm2d112 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu112 = ReLU(inplace=True)
        self.conv2d112 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x396):
        x397=self.batchnorm2d112(x396)
        x398=self.relu112(x397)
        x399=self.conv2d112(x398)
        return x399

m = M().eval()
x396 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x396)
end = time.time()
print(end-start)
