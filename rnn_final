device = 'choose_ur_device'

class RNNModel(nn.Module):
    def __init__(self, fit_size):
        super().__init__()
        self.rnn = nn.RNN(fit_size, fit_size * 2, 2, batch_first = True)
        self.lin1 = nn.Linear(fit_size * 2, fit_size * 2)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(fit_size * 2, fit_size)
        self.lin3 = nn.Linear(fit_size, 1)

    def forward(self, x):
        hidden, states = self.rnn(x)
        hidden = self.lin1(hidden)
        hidden = self.relu1(hidden)
        hidden = self.lin2(hidden)
        output = self.lin3(hidden)
        return output

rnn_model = RNNModel(fit_size)
rnn_model.to(device)

rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)

#fitting
rnn_ret_logs = train(rnn_model, rnn_optimizer, criterion, 30,  X_train, Y_train, X_test, Y_test)
