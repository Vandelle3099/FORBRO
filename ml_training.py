import os
import logging
from datetime import datetime
import pandas as pd
from ml_models import CryptoMLPredictor, ohlcv_to_dataframe

logger = logging.getLogger(__name__)

async def train_ml_model(symbol, timeframe, config, load_ohlcv, force_retrain=False):
    """Обучение ML-модели для прогнозирования тренда"""
    try:
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        model = CryptoMLPredictor(symbol, timeframe, model_dir)
        
        # Проверяем здоровье существующей модели
        if not force_retrain and model.check_model_health():
            logger.info(f"ML модель для {symbol} в хорошем состоянии, переобучение не требуется")
            return True
        
        # Загружаем исторические данные
        ohlcv = await load_ohlcv(symbol, timeframe, limit=240)  # Больше данных для обучения
        if not ohlcv or len(ohlcv) < 100:
            logger.warning(f"Недостаточно данных для обучения ML-модели {symbol}: {len(ohlcv) if ohlcv else 0} свечей")
            return False
        
        # Конвертируем в DataFrame
        df = ohlcv_to_dataframe(ohlcv)
        
        # Тестируем модель перед обучением
        if not model.test_model(df):
            logger.warning(f"Тест ML модели для {symbol} не пройден")
            return False
        
        # Обучаем модель
        result = model.train(df)
        accuracy = result['test_accuracy']
        logger.info(f"ML-модель для {symbol} обучена с точностью {accuracy:.2f}")
        
        # Проверяем точность
        if accuracy < config['min_ml_accuracy']:
            logger.warning(f"ML-модель для {symbol} имеет низкую точность: {accuracy:.2f} < {config['min_ml_accuracy']}")
            return False
            
        # Сохраняем модель
        model.save_model()
        logger.info(f"ML-модель для {symbol} успешно сохранена")
        
        return True
    
    except Exception as e:
        logger.error(f"Ошибка обучения ML-модели для {symbol}: {e}")
        return False

async def batch_train_models(symbols, timeframe, config, load_ohlcv):
    """Пакетное обучение ML-моделей для списка символов"""
    if not config['use_ml_predictions']:
        return
    
    logger.info(f"Начало пакетного обучения ML-моделей для {len(symbols)} символов")
    
    # Создаем директорию для моделей, если не существует
    os.makedirs('models', exist_ok=True)
    
    # Загружаем данные для всех символов
    symbol_data = {}
    for symbol in symbols:
        try:
            ohlcv = await load_ohlcv(symbol, timeframe, limit=240)  # Больше данных для обучения
            if ohlcv and len(ohlcv) >= 100:
                df = ohlcv_to_dataframe(ohlcv)
                symbol_data[symbol] = df
            else:
                logger.warning(f"Недостаточно данных для обучения ML-модели {symbol}")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных для ML-модели {symbol}: {e}")
    
    if not symbol_data:
        logger.warning("Нет данных для обучения ML-моделей")
        return
    
    # Обучаем модели пакетно
    results = {}
    for symbol, df in symbol_data.items():
        try:
            success = await train_ml_model(symbol, timeframe, config, load_ohlcv)
            results[symbol] = {'success': success}
        except Exception as e:
            logger.error(f"Ошибка обучения модели для {symbol}: {e}")
            results[symbol] = {'success': False, 'error': str(e)}
    
    # Выводим результаты
    successful = sum(1 for result in results.values() if result.get('success', False))
    logger.info(f"Завершено обучение ML-моделей: {successful}/{len(results)} успешно")
    
    return results 