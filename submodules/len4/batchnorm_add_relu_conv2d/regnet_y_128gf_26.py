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
        self.batchnorm2d75 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu96 = ReLU(inplace=True)
        self.conv2d124 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x390, x377):
        x391=self.batchnorm2d75(x390)
        x392=operator.add(x377, x391)
        x393=self.relu96(x392)
        x394=self.conv2d124(x393)
        return x394

m = M().eval()
x390 = torch.randn(torch.Size([1, 2904, 14, 14]))
x377 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x390, x377)
end = time.time()
print(end-start)
