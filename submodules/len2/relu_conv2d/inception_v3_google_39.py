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
        self.conv2d73 = Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)

    def forward(self, x249):
        x250=torch.nn.functional.relu(x249,inplace=True)
        x251=self.conv2d73(x250)
        return x251

m = M().eval()
x249 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x249)
end = time.time()
print(end-start)
