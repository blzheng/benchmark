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
        self.relu65 = ReLU(inplace=True)
        self.conv2d85 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)

    def forward(self, x267):
        x268=self.relu65(x267)
        x269=self.conv2d85(x268)
        return x269

m = M().eval()
x267 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x267)
end = time.time()
print(end-start)
