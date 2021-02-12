class CustomNet(nn.Module):
    def __init__(self, nb_class, params):
        super(CustomNet, self).__init__()

        self.conv_layers = []
        self.pool_layers = []

        next_input_channels = [1] 
        l_out = params['win_size']
        
        
        for layer in range(params['n_conv_layers']):
            
            output_channels = params['channels_' + str(layer)]
            kernel_size = params['conv_kernel_' + str(layer)] 
            stride = params['conv_stride_' + str(layer)] 
            
            self.conv_layers.append(nn.Conv2d(
                        in_channels=next_input_channels[-1], 
                        out_channels=output_channels, 
                        kernel_size=(1, kernel_size), 
                        stride=stride)
            )
            l_out = self.filter_size(l_out, kernel_size, stride)
            print(f'Conv: kernel {kernel_size}, stride {stride}, l_out: {l_out}')
            
            kernel_size = params['pool_kernel_' + str(layer)]
            stride = params['pool_stride_' + str(layer)]
            self.pool_layers.append(nn.MaxPool2d(
                        kernel_size=(1, kernel_size),
                        stride=stride)
            )
            l_out = self.filter_size(l_out, kernel_size, stride)

            next_input_channels.append(output_channels)  
            print(f'Pool: kernel {kernel_size}, stride {stride}, l_out: {l_out}')

        self.fc_input = l_out * output_channels
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(self.fc_input, params['nb_hidden'])
        self.fc2 = nn.Linear(params['nb_hidden'], nb_class)
        
    def forward(self, x):
        x = self.batch_norm(x)
        for c_layer, p_layer in zip(self.conv_layers, self.pool_layers):
            x = p_layer(F.relu(c_layer(x)))
    
        x = self.dropout(x)        
        x = x.view(-1, self.fc_input)
        x = F.relu(self.fc1(x))        
        x = F.dropout(x, training=self.training)
        x = F.softmax(self.fc2(x), 1)
        return x

    def filter_size(self, l_input, kernel_size, stride, dilatation=1):
        l_out = int((l_input - (kernel_size - 1) - 1)/stride + 1)
        return l_out