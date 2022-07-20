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
        self.conv2d49 = Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x177):
        x178=torch.nn.functional.relu(x177,inplace=True)
        x179=self.conv2d49(x178)
        return x179

m = M().eval()
x177 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x177)
end = time.time()
print(end-start)
