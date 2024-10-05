import math
from torch import nn
from torchvision.models import densenet201


class Illumination_classifier(nn.Module):
    def __init__(self, input_channels, init_weights=True):
        super(Illumination_classifier, self).__init__()

        self.features = densenet201(pretrained=False, progress=True)
        num_ftrs = self.features.classifier.in_features


        self.features.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(128, 2),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化权重

        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):

        activate = nn.PReLU()
        activate = activate.cuda()
        x = activate(self.features(x))
        return x
