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
        self.relu73 = ReLU(inplace=True)
        self.conv2d77 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d77 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x252):
        x253=self.relu73(x252)
        x254=self.conv2d77(x253)
        x255=self.batchnorm2d77(x254)
        return x255

m = M().eval()
x252 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x252)
end = time.time()
print(end-start)
