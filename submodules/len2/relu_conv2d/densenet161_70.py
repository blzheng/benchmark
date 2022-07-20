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
        self.relu71 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x254):
        x255=self.relu71(x254)
        x256=self.conv2d71(x255)
        return x256

m = M().eval()
x254 = torch.randn(torch.Size([1, 1152, 14, 14]))
start = time.time()
output = m(x254)
end = time.time()
print(end-start)
