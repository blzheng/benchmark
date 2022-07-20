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
        self.relu67 = ReLU(inplace=True)
        self.conv2d72 = Conv2d(1920, 1920, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)

    def forward(self, x233):
        x234=self.relu67(x233)
        x235=self.conv2d72(x234)
        return x235

m = M().eval()
x233 = torch.randn(torch.Size([1, 1920, 14, 14]))
start = time.time()
output = m(x233)
end = time.time()
print(end-start)
