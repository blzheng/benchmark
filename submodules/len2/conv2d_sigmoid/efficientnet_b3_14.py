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

    def forward(self, x220):
        x221=self.conv2d72(x220)
        x222=self.sigmoid14(x221)
        return x222

m = M().eval()
x220 = torch.randn(torch.Size([1, 34, 1, 1]))
start = time.time()
output = m(x220)
end = time.time()
print(end-start)
