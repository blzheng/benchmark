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
        self.relu100 = ReLU(inplace=True)
        self.conv2d104 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x342):
        x343=self.relu100(x342)
        x344=self.conv2d104(x343)
        return x344

m = M().eval()
x342 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x342)
end = time.time()
print(end-start)
