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
        self.relu46 = ReLU(inplace=True)
        self.conv2d50 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x162):
        x163=self.relu46(x162)
        x164=self.conv2d50(x163)
        return x164

m = M().eval()
x162 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x162)
end = time.time()
print(end-start)
