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
        self.conv2d326 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid58 = Sigmoid()
        self.conv2d327 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x1046, x1043):
        x1047=self.conv2d326(x1046)
        x1048=self.sigmoid58(x1047)
        x1049=operator.mul(x1048, x1043)
        x1050=self.conv2d327(x1049)
        return x1050

m = M().eval()
x1046 = torch.randn(torch.Size([1, 160, 1, 1]))
x1043 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1046, x1043)
end = time.time()
print(end-start)
