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
        self.relu13 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d14 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu14 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2d15 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu15 = ReLU(inplace=True)

    def forward(self, x45):
        x46=self.relu13(x45)
        x47=self.conv2d14(x46)
        x48=self.batchnorm2d14(x47)
        x49=self.relu14(x48)
        x50=self.conv2d15(x49)
        x51=self.batchnorm2d15(x50)
        x52=self.relu15(x51)
        return x52

m = M().eval()
x45 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x45)
end = time.time()
print(end-start)
