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
        self.conv2d164 = Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x522, x507):
        x523=operator.add(x522, x507)
        x524=self.conv2d164(x523)
        return x524

m = M().eval()
x522 = torch.randn(torch.Size([1, 256, 7, 7]))
x507 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x522, x507)
end = time.time()
print(end-start)
