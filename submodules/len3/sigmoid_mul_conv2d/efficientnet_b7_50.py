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
        self.sigmoid50 = Sigmoid()
        self.conv2d251 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x789, x785):
        x790=self.sigmoid50(x789)
        x791=operator.mul(x790, x785)
        x792=self.conv2d251(x791)
        return x792

m = M().eval()
x789 = torch.randn(torch.Size([1, 2304, 1, 1]))
x785 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x789, x785)
end = time.time()
print(end-start)
