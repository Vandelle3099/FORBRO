import numpy as np
from datetime import datetime, timedelta

def generate_test_ohlcv(symbol, timeframe='1h', num_candles=100):
    """Generate test OHLCV data"""
    now = datetime.now()
    timestamps = [(now - timedelta(hours=i)).timestamp() * 1000 for i in range(num_candles)]
    timestamps.reverse()
    
    # Generate realistic price movements
    base_price = 100 if 'BTC' not in symbol else 50000
    volatility = 0.02  # 2% volatility
    
    prices = [base_price]
    for _ in range(num_candles - 1):
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    ohlcv_data = []
    for i, timestamp in enumerate(timestamps):
        price = prices[i]
        high = price * (1 + abs(np.random.normal(0, volatility/2)))
        low = price * (1 - abs(np.random.normal(0, volatility/2)))
        volume = abs(np.random.normal(1000, 300))
        
        ohlcv_data.append([
            int(timestamp),  # timestamp
            float(price * (1 - volatility/4)),  # open
            float(high),  # high
            float(low),  # low
            float(price),  # close
            float(volume)  # volume
        ])
    
    return ohlcv_data

def get_test_symbols():
    """Get list of test symbols"""
    return [
        'BTC/USDT',
        'ETH/USDT',
        'BNB/USDT',
        'SOL/USDT',
        'DOGE/USDT',
        'XRP/USDT',
        'ADA/USDT',
        'AVAX/USDT',
        'MATIC/USDT',
        'LINK/USDT'
    ]

def get_test_ticker(symbol):
    """Get test ticker data"""
    base_prices = {
        'BTC/USDT': 50000,
        'ETH/USDT': 3000,
        'BNB/USDT': 500,
        'SOL/USDT': 150,
        'DOGE/USDT': 0.15,
        'XRP/USDT': 1.2,
        'ADA/USDT': 0.8,
        'AVAX/USDT': 35,
        'MATIC/USDT': 1.5,
        'LINK/USDT': 20
    }
    
    base_price = base_prices.get(symbol, 100)
    volatility = 0.02
    
    current_price = base_price * (1 + np.random.normal(0, volatility))
    volume = abs(np.random.normal(1000000, 300000))
    
    return {
        'symbol': symbol,
        'timestamp': datetime.now().timestamp() * 1000,
        'datetime': datetime.now().isoformat(),
        'high': current_price * (1 + volatility),
        'low': current_price * (1 - volatility),
        'bid': current_price * 0.9999,
        'ask': current_price * 1.0001,
        'last': current_price,
        'close': current_price,
        'baseVolume': volume,
        'quoteVolume': volume * current_price,
        'info': {}
    } 