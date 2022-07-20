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
        self.conv2d107 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()

    def forward(self, x338, x335):
        x339=self.conv2d107(x338)
        x340=self.sigmoid20(x339)
        x341=operator.mul(x340, x335)
        return x341

m = M().eval()
x338 = torch.randn(torch.Size([1, 726, 1, 1]))
x335 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x338, x335)
end = time.time()
print(end-start)