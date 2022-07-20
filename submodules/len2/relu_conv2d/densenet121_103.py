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
        self.relu104 = ReLU(inplace=True)
        self.conv2d104 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x371):
        x372=self.relu104(x371)
        x373=self.conv2d104(x372)
        return x373

m = M().eval()
x371 = torch.randn(torch.Size([1, 768, 7, 7]))
start = time.time()
output = m(x371)
end = time.time()
print(end-start)
