def train_func(model, optimizer, criterion, epochs, train, val, success):
  history = []
  acc_logs = []
  for i in range(epochs):
    model.train()
    sum = 0
    cnt = 0
    for batch, target in tqdm(train):
      # 1. # загружаем батч данных (вытянутый в линию)
      cnt += 1
      optimizer.zero_grad()
      x_batch = batch.to(device)
      #x_batch = x_batch.view(1, x_batch.shape[0])
      x_batch = x_batch.to(torch.float32)
      y_batch = target.to(device)
      y_batch = y_batch.to(torch.float32)

      # 2. вычисляем скор с помощью прямого распространения ( .forward or .__call__ )
      res = model(x_batch)

      # 3. вычислеяем - функцию потерь (loss)
      loss = criterion(res[0], y_batch)
      sum += loss
      #history.append(loss.item())

      # 4. вычисляем градиенты

      loss.backward()

      # 5. шаг градиентного спуска
      optimizer.step()
    sum /= cnt
    model.eval()
    acc = 0
    cnt = 0
    for batch, target in tqdm(val):
      # 1. # загружаем батч данных (вытянутый в линию)
      cnt += 1
      x_batch = batch.to(device)
      #x_batch = x_batch.view(1, x_batch.shape[0])
      x_batch = x_batch.to(torch.float32)
      y_batch = target.to(device)
      y_batch = y_batch.to(torch.float32)

      # 2. вычисляем скор с помощью прямого распространения ( .forward or .__call__ )
      res_check = model(x_batch)
      acc += get_acc(res_check, y_batch)
    acc /= cnt
    acc_logs.append(acc)
    print('Epoch:', i, 'Loss:', sum.item(), 'Accuracy:', acc )
    if acc < success:
      print('We are the champions')
      torch.save(model.state_dict(), 'weights_trade_model.pth')
      return acc_logs, history
    history.append(sum.item())

  return acc_logs, history
