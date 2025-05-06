import requests
import pandas as pd
from datetime import datetime, timedelta

def get_bybit_kline(symbol, start_ms, end_ms):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol.replace('/', ''),
        "interval": "60",
        "start": int(start_ms),
        "end": int(end_ms)
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data['retCode'] != 0:
            print(f"Ошибка для {symbol}: {data['retMsg']}")
            return pd.DataFrame()
        df = pd.DataFrame(data['result']['list'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    except Exception as e:
        print(f"Исключение для {symbol}: {e}")
        return pd.DataFrame()

start_dt = datetime(2025, 4, 25)
end_dt = datetime(2025, 4, 30)
ohlc_df = get_bybit_kline('BTC/USDT', 
                          start_dt.timestamp() * 1000, 
                          end_dt.timestamp() * 1000)
print(ohlc_df.head())