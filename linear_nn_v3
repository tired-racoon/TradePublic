device = 'choose_your_device'

lin_model = nn.Sequential(
  nn.Linear(fit_size, fit_size * 2),
  nn.BatchNorm1d(fit_size * 2),
  nn.LeakyReLU(0.2),
  #nn.ReLU(),
  #nn.ReLU(),
  nn.Linear(fit_size * 2, fit_size),
  #nn.BatchNorm1d(fit_size),
  nn.Linear(fit_size, 1)
)
lin_model.to(device)

lin_optimizer = torch.optim.Adam(lin_model.parameters(), lr=3e-4)

#training
lin_acc_logs, lin_ret_logs = train_func(lin_model, lin_optimizer, criterion, 20, train, val, 1)
