device = 'choose_ur_device'

class LSTMModel(nn.Module):
    def __init__(self, fit_size):
        super().__init__()
        self.lstm = nn.LSTM(fit_size, fit_size * 2, 2, batch_first = True)
        self.lin1 = nn.Linear(fit_size * 2, fit_size * 2)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(fit_size * 2, fit_size)
        self.lin3 = nn.Linear(fit_size, 1)

    def forward(self, x):
        hidden, states = self.lstm(x)
        hidden = self.lin1(hidden)
        hidden = self.relu1(hidden)
        hidden = self.lin2(hidden)
        output = self.lin3(hidden)
        return output
lstm_model = LSTMModel(fit_size)
lstm_model.to(device)

lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

#fitting
lstm_acc_logs, lstm_ret_logs = train_func(lstm_model, lstm_optimizer, criterion, 2000, train, val , 25)
