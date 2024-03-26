device = 'choose_ur_device'

class GRUModel(nn.Module):
    def __init__(self, fit_size):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(fit_size)
        self.gru = nn.GRU(fit_size, 10, 3, batch_first = True)
        self.norm2 = nn.BatchNorm1d(10)
        self.gru2 = nn.GRU(10, 10 , 3, batch_first = True)
        #self.lin1 = nn.Linear(fit_size * 2, fit_size * 2)
        self.lin2 = nn.Linear(10, 5)
        self.lin3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.norm1(x)
        hidden, states = self.gru(x)
        hidden = self.norm2(hidden)
        hidden, _ = self.gru2(hidden)
        #hidden = self.lin1(hidden)
        hidden = self.lin2(hidden)
        output = self.lin3(hidden)
        return output

gru_model = GRUModel(fit_size)
gru_model.to(device)

gru_optimizer = torch.optim.SGD(gru_model.parameters(), lr=3e-4)

#fitting
gru_acc_logs, gru_ret_logs = train_func(gru_model, gru_optimizer, criterion, 20, train, val, 1)
