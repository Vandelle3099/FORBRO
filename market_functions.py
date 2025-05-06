import ccxt.async_support as ccxt
import pandas as pd
import logging
import os
import pickle
import aiofiles
import aiohttp
import asyncio
from datetime import datetime, timedelta
from test_data import generate_test_ohlcv, get_test_symbols, get_test_ticker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Development mode flag
DEV_MODE = True

# Настройка Bybit
bybit = ccxt.bybit({
    'enableRateLimit': True,
    'defaultType': 'swap',
    'rateLimit': 100
})

async def load_markets():
    """Load available markets from Bybit"""
    try:
        logger.info("Loading markets...")
        
        if DEV_MODE:
            symbols = get_test_symbols()
            logger.info(f"Loaded {len(symbols)} test symbols")
            logger.info(f"Example pairs: {', '.join(symbols[:5])}")
            return symbols
            
        markets = await bybit.load_markets()
        symbols = [symbol for symbol, info in markets.items() if info['type'] == 'swap' and info['active'] and 
                  (symbol.endswith('/USDT:USDT') or symbol.endswith('/USDT')) and not 'BUSD' in symbol]
        
        # Преобразуем Bybit символы к стандартному формату
        standardized_symbols = [
            symbol.replace(':USDT', '') for symbol in symbols
        ]
        
        logger.info(f"Loaded {len(standardized_symbols)} active USDT futures pairs from Bybit")
        logger.info(f"Example pairs: {', '.join(standardized_symbols[:5])}")
        return standardized_symbols
    except Exception as e:
        logger.error(f"Error loading markets: {e}")
        return []

async def load_ohlcv(symbol, timeframe='1h', limit=60):
    """Load OHLCV data with caching"""
    try:
        logger.info(f"Loading OHLCV data for {symbol} ({timeframe})")
        cache_file = f'cache/{symbol.replace("/", "_")}_{timeframe}.pkl'
        os.makedirs('cache', exist_ok=True)
        
        if DEV_MODE:
            ohlcv = generate_test_ohlcv(symbol, timeframe, limit)
            logger.info(f"Generated {len(ohlcv)} test candles for {symbol}")
            return ohlcv
        
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age.total_seconds() < 3600:
                async with aiofiles.open(cache_file, 'rb') as f:
                    ohlcv = pickle.loads(await f.read())
                    logger.info(f"Loaded {len(ohlcv)} candles from cache for {symbol}")
                    return ohlcv

        logger.info(f"Fetching fresh OHLCV data for {symbol}")
        ohlcv = await bybit.fetch_ohlcv(symbol, timeframe, limit=limit)
        async with aiofiles.open(cache_file, 'wb') as f:
            await f.write(pickle.dumps(ohlcv))
        logger.info(f"Fetched and cached {len(ohlcv)} candles for {symbol}")
        return ohlcv
    except Exception as e:
        logger.error(f"Error loading OHLCV for {symbol}: {e}")
        if os.path.exists(cache_file):
            async with aiofiles.open(cache_file, 'rb') as f:
                ohlcv = pickle.loads(await f.read())
                logger.warning(f"Using old cache for {symbol} due to error")
                return ohlcv
        return None

async def fetch_fear_and_greed():
    """Get Fear & Greed Index"""
    try:
        logger.info("Fetching Fear & Greed Index...")
        
        if DEV_MODE:
            fng_value = 50
            logger.info(f"Using test Fear & Greed Index: {fng_value}")
            return fng_value
            
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.alternative.me/fng/') as response:
                data = await response.json()
                fng_value = int(data['data'][0]['value'])
                logger.info(f"Fear & Greed Index: {fng_value}")
                return fng_value
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed Index: {e}")
        return 50  # Fallback значение

async def filter_high_volume_symbols(symbols, min_volume_usd=1_000_000):
    """Filter symbols by trading volume"""
    try:
        logger.info(f"Filtering {len(symbols)} pairs by volume (min ${min_volume_usd/1_000_000:.2f}M)")
        cache_file = 'cache/high_volume_symbols.pkl'
        cache_max_age = 3600

        if DEV_MODE:
            # В режиме разработки возвращаем все тестовые символы
            volume_data = {symbol: 10_000_000 for symbol in symbols}
            logger.info(f"Using all {len(symbols)} test symbols as high volume")
            return symbols, volume_data

        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age.total_seconds() < cache_max_age:
                async with aiofiles.open(cache_file, 'rb') as f:
                    cached_data = pickle.loads(await f.read())
                    logger.info(f"Loaded volume filter cache: {len(cached_data)} pairs")
                    return cached_data, {}

        high_volume_symbols = []
        volume_data = {}
        
        async def check_volume(symbol):
            try:
                if DEV_MODE:
                    ticker = get_test_ticker(symbol)
                else:
                    ticker = await bybit.fetch_ticker(symbol)
                    
                volume_usd = ticker['baseVolume'] * ticker['close']
                
                logger.debug(f"Volume for {symbol}: ${volume_usd/1_000_000:.2f}M")
                
                if volume_usd > min_volume_usd:
                    logger.info(f"Pair {symbol}: Volume ${volume_usd/1_000_000:.2f}M (passed filter)")
                    return symbol, volume_usd
                logger.debug(f"Pair {symbol}: Volume ${volume_usd/1_000_000:.2f}M (filtered out)")
                return None, None
            except Exception as e:
                logger.error(f"Error getting volume for {symbol}: {e}")
                return None, None

        chunk_size = 5
        total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}/{total_chunks} ({len(chunk)} pairs)")
            results = await asyncio.gather(*[check_volume(symbol) for symbol in chunk])
            for symbol, volume_usd in results:
                if symbol:
                    high_volume_symbols.append(symbol)
                    volume_data[symbol] = volume_usd
            await asyncio.sleep(1)

        async with aiofiles.open(cache_file, 'wb') as f:
            await f.write(pickle.dumps(high_volume_symbols))
        
        logger.info(f"Filtering completed: {len(high_volume_symbols)} pairs passed volume filter")
        if high_volume_symbols:
            logger.info(f"Example high volume pairs: {', '.join(high_volume_symbols[:5])}")
        return high_volume_symbols, volume_data
        
    except Exception as e:
        logger.error(f"Error in filter_high_volume_symbols: {e}")
        return [], {} 