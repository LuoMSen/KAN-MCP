from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2=1, layers='256,128', dropout=0.3):
        super(MLP, self).__init__()

        self.all_layers = []
        layers = list(map(lambda x: int(x), layers.split(',')))
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        self.module = nn.Sequential(*self.all_layers)
        self.fc_out_1 = nn.Linear(layers[-1], output_dim1)
        self.fc_out_2 = nn.Linear(layers[-1], output_dim2)

    def forward(self, inputs):
        features = self.module(inputs)
        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        return features, emos_out, vals_out