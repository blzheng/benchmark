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
        self.conv2d71 = Conv2d(896, 224, kernel_size=(1, 1), stride=(1, 1))
        self.relu55 = ReLU()
        self.conv2d72 = Conv2d(224, 896, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid13 = Sigmoid()

    def forward(self, x224):
        x225=self.conv2d71(x224)
        x226=self.relu55(x225)
        x227=self.conv2d72(x226)
        x228=self.sigmoid13(x227)
        return x228

m = M().eval()
x224 = torch.randn(torch.Size([1, 896, 1, 1]))
start = time.time()
output = m(x224)
end = time.time()
print(end-start)
