# TradePublic
Public repository with some code for my TradeML project

Здесь находится код 2 версий парсера, несколько файлов, использованных при создании докер-контейнера (compose.yaml, Dockerfile, requirements), а также различные вспомогательные функции

Кроме того, есть 4 файла с кодом классов нейросетей (linear_nn, rnn, gru, lstm). Указаны версии в порядке их экспериментального исследования - было несколько попыток изменить архитектуру, оптимизатор, количество дней-фичей (параметр fit_size). Эти 4 модели (и соответственно их предыдущие версии, которые были удалены за ненадобностью) оказались неэффективны. 

В файле transformer.py лежит рабочая модель трансформера на keras, но он оказался не лучше свертки, однако в нем больше параметров, так что я оставил в качестве основной модели свертку, как более легковесную

В ноутбуке OrderSys лежат черновики функций, используемых для торговли. В файле trade_utils эти же функции уже скомпонованы в итоговый инструмент, который используется в app.py - это последняя версия приложения на сервере. Там используются потоки и кэширование, чтобы независимо от get-запросов внутри работал алгоритм, обновляющий прогнозы модели и осуществляющий торговлю.
