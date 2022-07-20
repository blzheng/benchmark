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
        self.relu22 = ReLU(inplace=True)
        self.conv2d27 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x87):
        x88=self.relu22(x87)
        x89=self.conv2d27(x88)
        return x89

m = M().eval()
x87 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
