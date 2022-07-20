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
        self.relu6 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x17, x25):
        x26=operator.add(x17, x25)
        x27=self.relu6(x26)
        x28=self.conv2d9(x27)
        return x28

m = M().eval()
x17 = torch.randn(torch.Size([1, 128, 28, 28]))
x25 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x17, x25)
end = time.time()
print(end-start)
