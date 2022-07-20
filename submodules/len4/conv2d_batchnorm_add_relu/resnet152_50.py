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
        self.conv2d147 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d147 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu142 = ReLU(inplace=True)

    def forward(self, x486, x490):
        x487=self.conv2d147(x486)
        x488=self.batchnorm2d147(x487)
        x491=operator.add(x488, x490)
        x492=self.relu142(x491)
        return x492

m = M().eval()
x486 = torch.randn(torch.Size([1, 512, 7, 7]))
x490 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x486, x490)
end = time.time()
print(end-start)
