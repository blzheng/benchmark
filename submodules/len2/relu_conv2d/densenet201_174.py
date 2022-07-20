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
        self.relu175 = ReLU(inplace=True)
        self.conv2d175 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x619):
        x620=self.relu175(x619)
        x621=self.conv2d175(x620)
        return x621

m = M().eval()
x619 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x619)
end = time.time()
print(end-start)
