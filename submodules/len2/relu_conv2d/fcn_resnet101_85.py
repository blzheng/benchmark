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
        self.relu85 = ReLU(inplace=True)
        self.conv2d90 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x297):
        x298=self.relu85(x297)
        x299=self.conv2d90(x298)
        return x299

m = M().eval()
x297 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x297)
end = time.time()
print(end-start)