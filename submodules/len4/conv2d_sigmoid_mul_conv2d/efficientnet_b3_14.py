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
        self.conv2d72 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()
        self.conv2d73 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x220, x217):
        x221=self.conv2d72(x220)
        x222=self.sigmoid14(x221)
        x223=operator.mul(x222, x217)
        x224=self.conv2d73(x223)
        return x224

m = M().eval()
x220 = torch.randn(torch.Size([1, 34, 1, 1]))
x217 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x220, x217)
end = time.time()
print(end-start)
