import torch
import torch.nn as nn
import MinkowskiEngine as ME

class EnergyRegressionModel (nn.Module):
    def __init__(self, **kwargs):
        super(EnergyRegressionModel, self).__init__(**kwargs)
        
        self.thing = 'a'

        self.model = nn.Sequential(ME.MinkowskiBatchNorm(num_features = 1),
                                   # ME.MinkowskiConvolution(in_channels = 1,
                                   #                         out_channels = 8,
                                   #                         kernel_size = 3,
                                   #                         stride = 1,
                                   #                         dimension = 3),
                                   # ME.MinkowskiConvolution(in_channels = 16,
                                   #                         out_channels = 16,
                                   #                         kernel_size = 3,
                                   #                         stride = 1,
                                   #                         dimension = 3),
                                   ResNetEncoder(in_features = 1,
                                                 depth = 8,
                                                 nFilters = 16),
                                   # ME.MinkowskiGlobalSumPooling(),
                                   # ME.MinkowskiGlobalAvgPooling(),
                                   ME.MinkowskiBatchNorm(num_features = 144),
                                   ME.MinkowskiLinear(144,
                                                      256),
                                   ME.MinkowskiReLU(),
                                   ME.MinkowskiLinear(256,
                                                      256),
                                   ME.MinkowskiReLU(),
                                   ME.MinkowskiLinear(256,
                                                      1),
                                   )

    def forward(self, x):
        return self.model(x)


class PositionEstimatorModel (nn.Module):
    def __init__(self, **kwargs):
        super(PositionEstimatorModel, self).__init__(**kwargs)
        
        self.thing = 'a'

        self.model = nn.Sequential(ME.MinkowskiBatchNorm(num_features = 1),
                                   # ME.MinkowskiConvolution(in_channels = 1,
                                   #                         out_channels = 8,
                                   #                         kernel_size = 3,
                                   #                         stride = 1,
                                   #                         dimension = 3),
                                   # ME.MinkowskiConvolution(in_channels = 16,
                                   #                         out_channels = 16,
                                   #                         kernel_size = 3,
                                   #                         stride = 1,
                                   #                         dimension = 3),
                                   ResNetEncoder(in_features = 1,
                                                 depth = 4,
                                                 nFilters = 16),
                                   # ME.MinkowskiGlobalSumPooling(),
                                   ME.MinkowskiBatchNorm(num_features = 80),
                                   ME.MinkowskiLinear(80,
                                                      256),
                                   ME.MinkowskiReLU(),
                                   ME.MinkowskiLinear(256,
                                                      256),
                                   ME.MinkowskiReLU(),
                                   ME.MinkowskiLinear(256,
                                                      # 4), # outputs are x, y, z, q
                                                      2), # outputs are z, q
                                   )

    def forward(self, x):
        return self.model(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, input):
        return input


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size, name = 'resBlock'):
        super(ResNetBlock, self).__init__()

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()
        
        self.norm1 = ME.MinkowskiBatchNorm(in_features)
        self.act1 = ME.MinkowskiReLU()
        self.conv1 = ME.MinkowskiConvolution(in_channels = in_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)

        self.norm2 = ME.MinkowskiBatchNorm(out_features)
        self.act2 = ME.MinkowskiReLU()
        self.conv2 = ME.MinkowskiConvolution(in_channels = out_features,
                                             out_channels = out_features,
                                             kernel_size = kernel_size,
                                             stride = 1,
                                             dimension = 3)
        
    def forward(self, x):

        residual = self.residual(x)
        
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        out += residual

        return out                                   
                                   

class ResNetEncoder(torch.nn.Module):
    def __init__(self, in_features, depth = 2, nFilters = 16, name='resnetencoder', **kwargs):
        super(ResNetEncoder, self).__init__()

        self.depth = depth # number of pool/unpool layers, not including input + output
        self.nFilters = nFilters
        self.in_features = in_features
        
        self.input_block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = self.in_features,
                out_channels = self.nFilters,
                kernel_size = 3,
                stride = 1,
                dimension = 3,
            ) 
        )

        # self.featureSizesEnc = [(self.nFilters*2**i, self.nFilters*2**(i+1))
        #                         for i in range(self.depth)]
        self.featureSizesEnc = [(self.nFilters*(i+1), self.nFilters*(i+2))
                                for i in range(self.depth)]
        
        self.encoding_layers = []

        self.encoding_blocks = []
        
        for i in range(self.depth):
            self.encoding_layers.append(
                ME.MinkowskiConvolution(
                    in_channels = self.featureSizesEnc[i][0],
                    out_channels = self.featureSizesEnc[i][1],
                    kernel_size = 2,
                    stride = 2,
                    dimension = 3)
            )
            # self.encoding_layers.append(
            #     nn.Sequential(
            #         ME.MinkowskiConvolution(
            #             in_channels = self.featureSizesEnc[i][0],
            #             out_channels = self.featureSizesEnc[i][0],
            #             kernel_size = 2,
            #             dimension = 3,),
            #         ME.MinkowskiSumPooling(kernel_size = 2,
            #                                stride = 2,
            #                                dimension = 3),
            #         )
            # )
            self.encoding_blocks.append(
                ResNetBlock(self.featureSizesEnc[i][1],
                            self.featureSizesEnc[i][1],
                            kernel_size = 3)
            )
        self.encoding_layers = nn.Sequential(*self.encoding_layers)
        self.encoding_blocks = nn.Sequential(*self.encoding_blocks)

    def forward(self, x):
        encodingFeatures = []
        coordKeys = []

        out = self.input_block(x)
        for i in range(self.depth):
            encodingFeatures.append(Identity()(out))
            coordKeys.append(out.coordinate_map_key)

            out = self.encoding_layers[i](out)
            out = self.encoding_blocks[i](out)

        return out
