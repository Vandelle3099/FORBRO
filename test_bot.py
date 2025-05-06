import asyncio
import logging
from crypto_signals_bot import (
    load_markets,
    filter_high_volume_symbols,
    calculate_indicators,
    analyze_signals,
    fetch_fear_and_greed,
    load_ohlcv
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_bot():
    try:
        logger.info("Starting bot test...")
        
        # Test market data loading
        symbols = await load_markets()
        logger.info(f"Loaded {len(symbols)} symbols")
        
        if not symbols:
            logger.error("Failed to load markets")
            return
        
        # Test volume filtering
        high_volume_symbols, volume_data = await filter_high_volume_symbols(symbols[:5])  # Test with first 5 symbols
        logger.info(f"Filtered {len(high_volume_symbols)} high volume symbols")
        
        if not high_volume_symbols:
            logger.error("No high volume symbols found")
            return
        
        # Test OHLCV data loading
        symbol = high_volume_symbols[0]
        ohlcv = await load_ohlcv(symbol)
        
        if not ohlcv:
            logger.error(f"Failed to load OHLCV data for {symbol}")
            return
            
        logger.info(f"Successfully loaded OHLCV data for {symbol}")
        
        # Test Fear & Greed Index
        fng = await fetch_fear_and_greed()
        logger.info(f"Fear & Greed Index: {fng}")
        
        # Test indicator calculation
        import pandas as pd
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        indicators = calculate_indicators(df, fng, symbol)
        if indicators:
            logger.info("Successfully calculated indicators")
        else:
            logger.error("Failed to calculate indicators")
            return
        
        # Test signal analysis
        volume_usd = volume_data.get(symbol, 0)
        price_change_1h = abs((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
        
        signal = await analyze_signals(symbol, indicators, df, volume_usd, price_change_1h)
        if signal:
            logger.info(f"Generated signal for {symbol}: {signal['type']}")
        else:
            logger.info(f"No signal generated for {symbol}")
        
        logger.info("Bot test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_bot()) 