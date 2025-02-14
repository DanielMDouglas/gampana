import torch.nn as nn

class EnergyRegressionModel (nn.Module):
    def __init__(self, **kwargs):
        super(EnergyRegressionModel, self).__init__(**kwargs)
        
        self.thing = 'a'

        self.model = nn.Sequential(nn.Conv3d(1, 2, 3),
                                   nn.Conv3d(2, 1, 3),
                                   )

    def forward(self, x):
        return self.model(x)
