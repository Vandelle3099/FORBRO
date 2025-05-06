import asyncio
import logging
from market_functions import load_markets, load_ohlcv
from ml_functions import train_ml_model, batch_train_models, get_ml_prediction

# Load config
import json
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Config file not found")
        return None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_ml():
    try:
        logger.info("Starting ML component test...")
        
        # Load config
        config = load_config()
        if not config:
            logger.error("Failed to load config")
            return
        
        # Test market data loading
        symbols = await load_markets()
        logger.info(f"Loaded {len(symbols)} symbols")
        
        if not symbols:
            logger.error("Failed to load markets")
            return
        
        # Test single model training
        test_symbol = symbols[0]
        success = await train_ml_model(
            symbol=test_symbol,
            timeframe=config['timeframe'],
            config=config,
            load_ohlcv=load_ohlcv
        )
        
        if success:
            logger.info(f"Successfully trained model for {test_symbol}")
            
            # Test prediction
            ohlcv = await load_ohlcv(test_symbol, config['timeframe'])
            if ohlcv:
                prediction = await get_ml_prediction(
                    symbol=test_symbol,
                    df=None,  # Let it load data internally
                    timeframe=config['timeframe'],
                    config=config,
                    load_ohlcv=load_ohlcv
                )
                if prediction:
                    logger.info(f"Got prediction for {test_symbol}: {prediction['direction']} with confidence {prediction['confidence']:.2f}")
                else:
                    logger.warning(f"No prediction available for {test_symbol}")
        else:
            logger.error(f"Failed to train model for {test_symbol}")
        
        # Test batch training with a small subset of symbols
        test_symbols = symbols[:3]  # Test with first 3 symbols
        results = await batch_train_models(
            symbols=test_symbols,
            timeframe=config['timeframe'],
            config=config,
            load_ohlcv=load_ohlcv
        )
        
        if results:
            successful = sum(1 for result in results.values() if result.get('success', False))
            logger.info(f"Batch training completed: {successful}/{len(results)} successful")
        else:
            logger.error("Batch training failed")
        
        logger.info("ML component test completed")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ml()) 