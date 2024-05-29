from tinkoff.invest import MoneyValue
from tinkoff.invest import OrderDirection, OrderType
from tinkoff.invest import Client, RequestError, PortfolioResponse, PositionsResponse, PortfolioPosition
import datetime
import pytz


data_to_display = {'info': 'None </br>'}
# a function to get portfolio and operations with an account
def get_portf(acc_id, token):
    try:
        with Client(token) as client:
            resp: PortfolioResponse = client.operations.get_portfolio(account_id=acc_id)
            opers = client.operations.get_portfolio(account_id=acc_id)
            return resp, opers

    except RequestError as e:
        print(str(e))


# convert MoneyValue to rub
def cast_money(v):
    return v.units + v.nano / 1e9  # nano - 9 нулей

def dict_cast_money(v):
    return v['units'] + v['nano'] / 1e9  # nano - 9 нулей

def portfolio_pose_todict(p: PortfolioPosition):
    r = {
        'figi': p.figi,
        'quantity': cast_money(p.quantity),
        'expected_yield': cast_money(p.expected_yield),
        'instrument_type': p.instrument_type,
        'average_buy_price': cast_money(p.average_position_price),
        'currency': p.average_position_price.currency,
        'nkd': cast_money(p.current_nkd),
    }

    r['sell_sum'] = (r['average_buy_price'] * r['quantity']) + r['expected_yield'] + (r['nkd'] * r['quantity'])
    r['comission'] = r['sell_sum'] * 0.003
    r['tax'] = r['expected_yield'] * 0.013 if r['expected_yield'] > 0 else 0

    return r

def get_acc_id(token):
    with Client(token) as client:
        r = client.users.get_accounts()
    ids = []
    for elem in r.accounts:
        ids.append(elem.id)
    return ids


# a useful wrapper on post_order function
def trade_stock(df, acc_id, token, quantity, price, direction, ticker=None, figi=None ):
    if ticker is None and figi is None:
        raise KeyError('Ticker or figi must be provided')
    # Определяем тип сделки: покупка или продажа
    if direction == 'buy':
        direct = OrderDirection.ORDER_DIRECTION_BUY
    elif direction == 'sell':
        direct = OrderDirection.ORDER_DIRECTION_SELL
    else:
        raise KeyError('Direction must be "buy" or "sell"')
    # Получаем figi по тикеру
    if figi is None:
        try:
            figi = df[df['ticker'] == ticker]['figi'].values[0]
        except:
            raise KeyError('Ticker not in Companies DF')
    else:
        figi = figi
    # Получаем значение цены в терминах Tinkoff API
    deal_price = MoneyValue(units=int(price), nano=int(1e9 * (price - int(price))))

    with Client(token) as client:
        resp = client.orders.post_order(figi=figi, quantity=quantity, price=deal_price,
                                        direction=direct, account_id=acc_id, order_type=OrderType.ORDER_TYPE_LIMIT)
    return resp


# a function to get last prices for a DICT of actives
def get_prices_for_actives(df, actives, token):
    prices = dict()
    figs = [stock['figi'] for stock in actives]
    with Client(token) as client:
        # for stock in actives
        u = client.market_data.get_last_prices(figi=figs)
    prices['time'] = datetime.datetime.now(pytz.utc)
    for resp in u.last_prices:
        price = cast_money(resp.price)
        lot = df[df['figi'] == resp.figi]['lot'].item()
        prices[resp.figi] = {'price': price, 'lot': lot, 'total': price * lot}
    return prices


def check_stocks_to_sell(df, actives, acc_id, token, predictions):
    global data_to_display
    if not len(actives):
        return []
    prices = get_prices_for_actives(df, actives, token)
    data_to_display['info'] = 'Prices for actives are got </br>'
    order_responses = []
    for stock in actives:
        change = 100 * abs(prices[stock['figi']]['price'] - stock['average_buy_price']) / stock['average_buy_price']
        try:
            ticker = df[df['figi'] == stock['figi']]['ticker'].values[0]
        except:
            raise KeyError("Can't get ticker via figi")
        if change > 3 or predictions[stock['figi']] <= -80:
            try:
                resp = trade_stock(df, acc_id, token, stock['quantity'], prices[stock['figi']]['price'], 'sell',
                                   figi=stock['figi'])
                order_responses.append(resp)
            except:
                raise ValueError('Something went wrong while selling a stock')

    return order_responses

def preds_to_orders(df, predictions, acc_id, token, currencies):  # predictions - a dict like "figi" :  prediction
    global data_to_display
    to_buy = []
    total = 0
    if not len(predictions):
        return []
    for pair in predictions:
        if predictions[pair] >= 80:
            total += predictions[pair]
            to_buy.append({'figi': pair, 'prediction': predictions[pair]})
    if not total:
        return []
    to_buy.sort(key=lambda x: x['prediction'], reverse=True)
    data_to_display['info'] = 'Sorted list of stocks for purchase is done </br>'
    for i, stock in enumerate(to_buy):
        to_buy[i]['weight'] = stock['prediction'] / total

    # CHECKPOINT 1
    # return to_buy
    free_balance = 0
    for curr in currencies:
        if curr['figi'] == 'RUB000UTSTOM':
            free_balance = curr['quantity']
    if free_balance <= 0:
        print('Balance equals to zero or less')
        return None
    free_balance *= 0.7
    data_to_display['info'] = 'Free balance is got </br>'
    cur_prices = get_prices_for_actives(df, to_buy, token)
    rest_balance = free_balance
    # CHECKPOINT 2
    # return cur_prices

    for i, stock in enumerate(to_buy):
        total = cur_prices[stock['figi']]['total']
        fraction = free_balance * stock['weight']
        quantity = int(fraction / total)
        if quantity * total <= rest_balance:
            to_buy[i]['quantity'] = quantity
            rest_balance -= quantity * total
        else:
            to_buy[i]['quantity'] = 0
    data_to_display['info'] = 'Operations are done! </br>'
    for stock in to_buy:
        try:
            ticker = df[df['figi'] == stock['figi']]['ticker'].values[0]
        except:
            raise KeyError("Can't get ticker via figi")
        data_to_display['info'] += ticker + ' ' +  str(stock['quantity']) + '</br>'
    # CHECKPOINT 3
    # return to_buy
    # TODO create an algorithm to choose price better
    buy_resps = []
    for stock in to_buy:
        try:
            resp = trade_stock(df, acc_id, token, stock['quantity'], cur_prices[stock['figi']]['price'], 'buy',
                               figi=stock['figi'])
            buy_resps.append(resp)
        except:
            raise ValueError('Something went wrong while buying a stock')
    return buy_resps
