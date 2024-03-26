!pip install tinkoff-investments
import pandas as pd
from tinkoff.invest import Client, InstrumentStatus, SharesResponse, InstrumentIdType
from tinkoff.invest.services import InstrumentsService, MarketDataService
import os
import datetime
from tinkoff.invest import CandleInterval

from tinkoff.invest import Client

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
import nltk

from collections import Counter
from typing import List
import random

import seaborn
seaborn.set(palette='summer')

token = YOUR_TOKEN

df = pd.read_csv('./companies_fixed.csv')
df['cap'] = df['cap'].astype(float)

appropriate_years = list(i for i in range(2004, 2024))
#appropriate_years = [2020, 2021, 2022, 2023]
random.shuffle(appropriate_years)



# Class for preprocessing data

class DataProcessing:
    def __init__(self, percents, train):
        self.data = percents
        self.train = train
        self.i = int(self.train * len(self.data))
        self.stock_train = self.data[0: self.i]
        self.stock_test = self.data[self.i:]
        self.input_train = []
        self.output_train = []
        self.input_test = []
        self.output_test = []

    def gen_train(self, seq_len, trhd, trhd_del):
        """
        Generates training data
        :param seq_len: length of window
        :return: X_train and Y_train
        """
        for i in range((len(self.stock_train)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.stock_train[i: i + seq_len], np.double)
            y = np.array([self.stock_train[i + seq_len + 1]], np.double)
            flag = 1
            last_elem = 0
            prev_el = 0
            for i, elem in enumerate(x):
              if i == 0:
                x[i] = elem
              elif i == 1:
                x[i] = (elem + last_elem) / 2
              else:
                x[i] = (elem + last_elem + prev_el) / 3
              if abs(elem) > trhd_del:
                flag = 0
                break
              if x[i] > trhd:
                 x[i] = trhd
              if x[i] < -trhd:
                x[i] = -trhd
              prev_el = last_elem
              last_elem = elem
            for i, elem in enumerate(y):
              y[i] = (elem + last_elem + prev_el)/3
              if abs(elem) > trhd_del or not flag:
                flag = 0
                break
              if y[i] > trhd:
                 y[i] = trhd
              if y[i] < -trhd:
                y[i] = -trhd
            if flag:
              self.input_train.append(x)
              self.output_train.append(y)
        self.X_train = np.array(self.input_train)
        self.Y_train = np.array(self.output_train)

    def gen_test(self, seq_len, trhd, trhd_del):
        """
        Generates test data
        :param seq_len: Length of window
        :return: X_test and Y_test
        """
        for i in range((len(self.stock_test)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.stock_test[i: i + seq_len], np.double)
            y = np.array([self.stock_test[i + seq_len + 1]], np.double)
            flag = 1
            last_elem = 0
            prev_el = 0
            for i, elem in enumerate(x):
              if i == 0:
                x[i] = elem
              elif i == 1:
                x[i] = (elem + last_elem) / 2
              else:
                x[i] = (elem + last_elem + prev_el) / 3
              if abs(elem) > trhd_del:
                flag = 0
                break
              if x[i] > trhd:
                 x[i] = trhd
              if x[i] < -trhd:
                x[i] = -trhd
              prev_el = last_elem
              last_elem = elem
            for i, elem in enumerate(y):
              y[i] = (elem + last_elem + prev_el)/3
              if abs(elem) > trhd_del or not flag:
                flag = 0
                break
              if y[i] > trhd:
                 y[i] = trhd
              if y[i] < -trhd:
                y[i] = -trhd
            if flag:
              self.input_test.append(x)
              self.output_test.append(y)
        self.X_test = np.array(self.input_test)
        self.Y_test = np.array(self.output_test)

#class for final dataset
class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

#function to get train and test data
def get_train_test(ticker, appropriate_years, trhd, trhd_del):
  prices = []


  with Client(token) as client:
    for year in appropriate_years:
      try:
        r = client.market_data.get_candles(figi = df[df['ticker'] == ticker]['figi'].item(), from_ = datetime.datetime(year, 1, 1), to =  datetime.datetime(year, 12, 31), interval = CandleInterval.CANDLE_INTERVAL_DAY)
        cop = pd.DataFrame(r.candles)
        for d in cop['close']:
          prices.append(float(d['units']) + float('0.' + str(d['nano'])))
      except:
        print('Skipped: ', year, ticker, df[df['ticker'] == ticker]['figi'].item())
        continue

  percents = []
  if not prices:
    return np.array([]), np.array([]), np.array([]), np.array([])
  last_price = prices[0]
  for price in prices[1:]:
    if price != last_price:
      percents.append(round((price - last_price) * 100 *100 / last_price, 2))
    last_price = price
  new_set = DataProcessing(percents, 0.9)
  new_set.gen_train(fit_size, trhd, trhd_del)
  new_set.gen_test(fit_size, trhd, trhd_del)
  X_train = new_set.X_train.reshape((new_set.X_train.shape[0], fit_size))
  Y_train = new_set.Y_train

  X_test = new_set.X_test.reshape(new_set.X_test.shape[0], fit_size)
  Y_test = new_set.Y_test
  return X_train, Y_train, X_test, Y_test

# get an example of data
def get_example(model, val):
  for batch, target in val:
      # 1. # загружаем батч данных (вытянутый в линию)

      x_batch = batch.to(device)
      #x_batch = x_batch.view(1, x_batch.shape[0])
      x_batch = x_batch.to(torch.float32)
      y_batch = target.to(device)
      y_batch = y_batch.to(torch.float32)

      # 2. вычисляем скор с помощью прямого распространения ( .forward or .__call__ )
      res_check = model(x_batch)
      return res_check.cpu().detach().numpy(), y_batch.cpu().detach().numpy()
      break


# Getting all data
X = np.array([]).reshape(0, fit_size)
Y = np.array([]).reshape(0, 1)          # list(df[df['cap'] > 700]['ticker'])
for ticker in list(df[df['cap'] > 300]['ticker']):                          #replace the list with your list of tickers if this is needed
  X_tr, Y_tr, X_val, Y_val = get_train_test(ticker, appropriate_years, 200, 300)
  if X_tr.shape[0] and Y_val.shape[0] and Y_tr.shape[0] and X_val.shape[0]:
    X = np.concatenate((X, X_tr), 0)
    Y = np.concatenate((Y, Y_tr), 0)
    X = np.concatenate((X, X_val), 0)
    Y = np.concatenate((Y, Y_val), 0)

dataset = pd.DataFrame(X, columns=list('x{0}'.format(i) for i in range(X.shape[1])))
dataset['label'] = Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

train = MyDataset(X_train, Y_train)
val = MyDataset(X_test, Y_test)
train = DataLoader(train, batch_size=batch_size, shuffle=True)
val = DataLoader(val, batch_size=batch_size, shuffle=False)
