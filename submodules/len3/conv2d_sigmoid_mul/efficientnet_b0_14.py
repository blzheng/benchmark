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
        self.conv2d73 = Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()

    def forward(self, x221, x218):
        x222=self.conv2d73(x221)
        x223=self.sigmoid14(x222)
        x224=operator.mul(x223, x218)
        return x224

m = M().eval()
x221 = torch.randn(torch.Size([1, 48, 1, 1]))
x218 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x221, x218)
end = time.time()
print(end-start)
