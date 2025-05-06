import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import concurrent.futures
import numpy as np
import hmac
import hashlib
import logging
import os
import random
import yfinance as yf
import pytz
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Константы
API_KEY = "uZAnDeM0xtFoJuHMKx"
API_SECRET = None  # Заглушка для API Secret
INPUT_FILE = 'filtered_results.xlsx'
OUTPUT_FILE = 'signals_history_with_win.xlsx'
MAX_WORKERS = 4  # Максимальное количество параллельных потоков

# Расширенный список мем-коинов
MEME_COINS = ['SUNDOG/USDT', 'DOGS/USDT', 'PNUT/USDT', 'PEPE/USDT', 'SHIB/USDT', 'DOGE/USDT', 'FLOKI/USDT', 'BONK/USDT', 
              'WIF/USDT', 'MEME/USDT', 'CAT/USDT', 'POPCAT/USDT', 'GALA/USDT']

# Расширенный словарь резервных пар
FALLBACK_PAIRS = {
    # Основные пары используют ETH/USDT или BTC/USDT как запасную
    'BTC/USDT': 'ETH/USDT',
    'ETH/USDT': 'BTC/USDT',
    # Альткоины используют ETH/USDT
    'ALCH/USDT': 'ETH/USDT',
    'ALEO/USDT': 'ETH/USDT',
    'AVAX/USDT': 'ETH/USDT',
    'SOL/USDT': 'ETH/USDT',
    'ADA/USDT': 'ETH/USDT',
    'DOT/USDT': 'ETH/USDT',
    'XRP/USDT': 'ETH/USDT',
    'LINK/USDT': 'ETH/USDT',
    'MATIC/USDT': 'ETH/USDT',
    'UNI/USDT': 'ETH/USDT',
    'AAVE/USDT': 'ETH/USDT',
    'SNX/USDT': 'ETH/USDT',
    'SUSHI/USDT': 'ETH/USDT',
    'COMP/USDT': 'ETH/USDT',
    'YFI/USDT': 'ETH/USDT',
    'MKR/USDT': 'ETH/USDT',
    'LTC/USDT': 'ETH/USDT',
    'BCH/USDT': 'ETH/USDT',
    'ETC/USDT': 'ETH/USDT',
    'ZEC/USDT': 'ETH/USDT',
    'DASH/USDT': 'ETH/USDT',
    'XMR/USDT': 'ETH/USDT',
    'ZRX/USDT': 'ETH/USDT',
    'BAT/USDT': 'ETH/USDT',
    'ATOM/USDT': 'ETH/USDT',
    'ALGO/USDT': 'ETH/USDT',
    'ICX/USDT': 'ETH/USDT',
    'ONT/USDT': 'ETH/USDT',
    'IOTA/USDT': 'ETH/USDT',
    'ZIL/USDT': 'ETH/USDT',
    'INJ/USDT': 'ETH/USDT',
    'FET/USDT': 'ETH/USDT',
    'OGN/USDT': 'ETH/USDT',
    'CTK/USDT': 'ETH/USDT',
    'NEAR/USDT': 'ETH/USDT',
    'RUNE/USDT': 'ETH/USDT',
    'OCEAN/USDT': 'ETH/USDT',
    'CRV/USDT': 'ETH/USDT',
    'FIL/USDT': 'ETH/USDT',
    'EGLD/USDT': 'ETH/USDT',
    'KAVA/USDT': 'ETH/USDT',
    'LRC/USDT': 'ETH/USDT',
    'ALPHA/USDT': 'ETH/USDT',
    'SXP/USDT': 'ETH/USDT',
    
    # Мем-коины используют DOGE/USDT
    'SUNDOG/USDT': 'DOGE/USDT',
    'DOGS/USDT': 'DOGE/USDT',
    'PNUT/USDT': 'DOGE/USDT',
    'PEPE/USDT': 'DOGE/USDT',
    'FLOKI/USDT': 'DOGE/USDT',
    'BONK/USDT': 'DOGE/USDT',
    'WIF/USDT': 'DOGE/USDT',
    'MEME/USDT': 'DOGE/USDT',
    'CAT/USDT': 'DOGE/USDT',
    'POPCAT/USDT': 'DOGE/USDT',
    'SHIB/USDT': 'DOGE/USDT',
    
    # Остальные используют BNB/USDT
    'CAKE/USDT': 'BNB/USDT',
    'BNB/USDT': 'ETH/USDT',
    'GALA/USDT': 'BNB/USDT',
    'AXS/USDT': 'BNB/USDT',
    'SAND/USDT': 'BNB/USDT',
    'MANA/USDT': 'BNB/USDT',
    'ENJ/USDT': 'BNB/USDT',
    'CHZ/USDT': 'BNB/USDT',
    'ONE/USDT': 'BNB/USDT',
    'THETA/USDT': 'BNB/USDT',
    'VET/USDT': 'BNB/USDT',
    'TRX/USDT': 'BNB/USDT',
    'EOS/USDT': 'BNB/USDT',
    'NEO/USDT': 'BNB/USDT',
    'QTUM/USDT': 'BNB/USDT',
    'XLM/USDT': 'BNB/USDT',
    'KSM/USDT': 'BNB/USDT',
    'BB/USDT': 'BNB/USDT',
    
    # BUSD пары используют свои USDT аналоги
    'BTC/BUSD': 'BTC/USDT',
    'ETH/BUSD': 'ETH/USDT',
    'BNB/BUSD': 'BNB/USDT',
    'ADA/BUSD': 'ADA/USDT',
    'XRP/BUSD': 'XRP/USDT',
    'DOGE/BUSD': 'DOGE/USDT',
    'SOL/BUSD': 'SOL/USDT',
    'DOT/BUSD': 'DOT/USDT',
    'MATIC/BUSD': 'MATIC/USDT',
    'LINK/BUSD': 'LINK/USDT',
    'UNI/BUSD': 'UNI/USDT',
    'AAVE/BUSD': 'AAVE/USDT',
    'SXP/BUSD': 'BNB/USDT',
    'LUNA/BUSD': 'BNB/USDT',
    
    # Значение по умолчанию
    'DEFAULT': 'BNB/USDT'
}

def excel_date_to_datetime(excel_date):
    """Convert Excel date to Python datetime"""
    try:
        return pd.Timestamp('1899-12-30') + pd.Timedelta(days=float(excel_date))
    except:
        return pd.NaT

def format_symbol_for_api(symbol):
    """Format symbol for Bybit API (remove / and ensure uppercase)"""
    return symbol.replace('/', '').upper()

def filter_anomalies(df):
    """Фильтрует аномальные данные из DataFrame"""
    logger.info("==== НАЧАЛО ФИЛЬТРАЦИИ АНОМАЛИЙ ====")
    
    if df is None or df.empty:
        logger.warning("ОПЕРАЦИЯ: Получен пустой DataFrame для фильтрации аномалий")
        return pd.DataFrame()
    
    logger.info(f"ОПЕРАЦИЯ: Начинаем фильтрацию DataFrame с {len(df)} строками")
    if 'symbol' in df.columns:
        unique_symbols_before = df['symbol'].unique()
        logger.info(f"ОПЕРАЦИЯ: В начале фильтрации {len(unique_symbols_before)} уникальных пар")
        logger.info(f"ОПЕРАЦИЯ: Примеры пар до фильтрации: {', '.join(sorted(unique_symbols_before)[:10])}")
    
    # Копируем DataFrame для фильтрации
    logger.info("ОПЕРАЦИЯ: Создание копии DataFrame для фильтрации")
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    # Проверяем наличие необходимых колонок
    required_columns = ['symbol', 'type', 'entry', 'tp', 'sl', 'datetime']
    logger.info(f"ОПЕРАЦИЯ: Проверка наличия необходимых колонок: {', '.join(required_columns)}")
    missing_columns = [col for col in required_columns if col not in filtered_df.columns]
    
    if missing_columns:
        logger.error(f"ОПЕРАЦИЯ: Отсутствуют необходимые колонки: {', '.join(missing_columns)}")
        return pd.DataFrame()
    
    # Фильтрация некорректных значений
    logger.info("ОПЕРАЦИЯ: Начинаем фильтрацию аномалий...")
    
    # 1. Удаляем строки с пустыми значениями
    logger.info("ОПЕРАЦИЯ: Удаление строк с пустыми значениями")
    initial_count = len(filtered_df)
    filtered_df = filtered_df.dropna(subset=required_columns)
    na_removed = initial_count - len(filtered_df)
    logger.info(f"ОПЕРАЦИЯ: Удалено {na_removed} строк с пустыми значениями")
    
    # 2. Проверяем тип сигнала
    logger.info("ОПЕРАЦИЯ: Проверка корректности типа сигнала (long/short)")
    initial_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['type'].isin(['long', 'short'])]
    type_removed = initial_count - len(filtered_df)
    logger.info(f"ОПЕРАЦИЯ: Удалено {type_removed} строк с некорректным типом сигнала")
    
    # Преобразуем числовые колонки в float, если они еще не преобразованы
    numeric_columns = ['entry', 'tp', 'sl']
    logger.info(f"ОПЕРАЦИЯ: Преобразование числовых колонок в float: {', '.join(numeric_columns)}")
    for col in numeric_columns:
        if filtered_df[col].dtype != 'float64':
            try:
                filtered_df[col] = filtered_df[col].astype(float)
                logger.info(f"ОПЕРАЦИЯ: Колонка {col} успешно преобразована в float")
            except Exception as e:
                logger.error(f"ОПЕРАЦИЯ: Ошибка при преобразовании колонки {col} в float: {e}")
                sample_values = filtered_df[col].head(5).tolist()
                logger.error(f"ОПЕРАЦИЯ: Примеры значений в колонке {col}: {sample_values}")
                return pd.DataFrame()
    
    # 3. Удаляем строки с отрицательными ценами
    logger.info("ОПЕРАЦИЯ: Проверка на отрицательные цены")
    initial_count = len(filtered_df)
    filtered_df = filtered_df[(filtered_df['entry'] > 0) & (filtered_df['tp'] > 0) & (filtered_df['sl'] > 0)]
    negative_removed = initial_count - len(filtered_df)
    logger.info(f"ОПЕРАЦИЯ: Удалено {negative_removed} строк с отрицательными ценами")
    
    # 4. Проверяем корректность TP и SL относительно типа сигнала
    logger.info("ОПЕРАЦИЯ: Проверка корректности TP и SL относительно типа сигнала")
    initial_count = len(filtered_df)
    long_mask = (filtered_df['type'] == 'long') & (filtered_df['tp'] > filtered_df['entry']) & (filtered_df['sl'] < filtered_df['entry'])
    short_mask = (filtered_df['type'] == 'short') & (filtered_df['tp'] < filtered_df['entry']) & (filtered_df['sl'] > filtered_df['entry'])
    
    filtered_df = filtered_df[long_mask | short_mask]
    tpsl_removed = initial_count - len(filtered_df)
    logger.info(f"ОПЕРАЦИЯ: Удалено {tpsl_removed} строк с некорректными TP/SL относительно типа сигнала")
    
    # 5. Проверяем, что TP и SL не слишком близко к цене входа
    min_distance_percent = 0.001  # Минимальное расстояние 0.1%
    logger.info(f"ОПЕРАЦИЯ: Проверка минимального расстояния TP/SL от цены входа (мин. {min_distance_percent*100}%)")
    
    initial_count = len(filtered_df)
    
    # Для long позиций: TP должен быть выше entry на min_distance_percent, SL должен быть ниже entry на min_distance_percent
    long_valid = (filtered_df['type'] == 'long') & \
                 ((filtered_df['tp'] - filtered_df['entry']) / filtered_df['entry'] >= min_distance_percent) & \
                 ((filtered_df['entry'] - filtered_df['sl']) / filtered_df['entry'] >= min_distance_percent)
    
    # Для short позиций: TP должен быть ниже entry на min_distance_percent, SL должен быть выше entry на min_distance_percent
    short_valid = (filtered_df['type'] == 'short') & \
                  ((filtered_df['entry'] - filtered_df['tp']) / filtered_df['entry'] >= min_distance_percent) & \
                  ((filtered_df['sl'] - filtered_df['entry']) / filtered_df['entry'] >= min_distance_percent)
    
    filtered_df = filtered_df[long_valid | short_valid]
    distance_removed = initial_count - len(filtered_df)
    logger.info(f"ОПЕРАЦИЯ: Удалено {distance_removed} строк с TP/SL слишком близко к цене входа")
    
    # Итоги фильтрации
    total_removed = original_count - len(filtered_df)
    logger.info(f"ОПЕРАЦИЯ: Итоги фильтрации: всего удалено {total_removed} строк ({total_removed/original_count*100:.1f}% от исходных данных)")
    logger.info(f"ОПЕРАЦИЯ: Осталось {len(filtered_df)} строк после фильтрации")
    
    # ДИАГНОСТИКА: Количество пар после фильтрации
    if 'symbol' in filtered_df.columns:
        unique_symbols_after = filtered_df['symbol'].unique()
        logger.info(f"ОПЕРАЦИЯ: После фильтрации осталось {len(unique_symbols_after)} уникальных пар")
        
        if len(unique_symbols_after) < len(unique_symbols_before):
            missing_symbols = set(unique_symbols_before) - set(unique_symbols_after)
            logger.warning(f"ОПЕРАЦИЯ: {len(missing_symbols)} пар полностью отфильтрованы:")
            missing_list = sorted(list(missing_symbols))
            for i in range(0, len(missing_list), 10):
                batch = missing_list[i:i+10]
                logger.warning(f"  - Отфильтрованные пары {i+1}-{i+len(batch)}: {', '.join(batch)}")
        
        logger.info(f"ОПЕРАЦИЯ: Примеры оставшихся пар: {', '.join(sorted(unique_symbols_after)[:10])}")
        
        # Подсчитываем сигналы по каждой паре
        signals_by_symbol = filtered_df['symbol'].value_counts()
        logger.info(f"ОПЕРАЦИЯ: Статистика по сигналам на пару после фильтрации:")
        logger.info(f"  - Минимальное количество сигналов: {signals_by_symbol.min()}")
        logger.info(f"  - Максимальное количество сигналов: {signals_by_symbol.max()}")
        logger.info(f"  - Среднее количество сигналов: {signals_by_symbol.mean():.1f}")
        
        # Выводим количество сигналов для первых 10 пар
        top_pairs = signals_by_symbol.nlargest(10)
        logger.info(f"ОПЕРАЦИЯ: Количество сигналов для топ-10 пар:")
        for symbol, count in top_pairs.items():
            logger.info(f"  - {symbol}: {count} сигналов")
    
    logger.info("==== ЗАВЕРШЕНИЕ ФИЛЬТРАЦИИ АНОМАЛИЙ ====")
    return filtered_df

def get_bybit_kline(symbol, start_ms, end_ms, retries=3):
    """Get OHLC data from Bybit public API"""
    url = "https://api.bybit.com/v5/market/kline"
    formatted_symbol = format_symbol_for_api(symbol)
    
    params = {
        "category": "linear",
        "symbol": formatted_symbol,
        "interval": "60",  # 1 hour
        "start": int(start_ms),
        "end": int(end_ms),
        "limit": 200
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('retCode', -1) != 0:
                logger.warning(f"Ошибка для {symbol}: {data.get('retMsg', 'Unknown error')}")
                if attempt < retries - 1:
                    logger.debug(f"Попытка {attempt + 2}/{retries}...")
                    time.sleep(1)
                    continue
                return pd.DataFrame()
            
            if not data.get('result', {}).get('list'):
                logger.warning(f"Нет данных для {symbol}")
                return pd.DataFrame()
                
            # Parse data into DataFrame
            df = pd.DataFrame(data['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            df = df.sort_values('timestamp')
            
            logger.info(f"Получено {len(df)} свечей для {symbol}")
            return df
            
        except Exception as e:
            logger.warning(f"Исключение для {symbol}: {e}")
            if attempt < retries - 1:
                logger.debug(f"Попытка {attempt + 2}/{retries}...")
                time.sleep(1)
            else:
                return pd.DataFrame()

def get_bybit_kline_auth(symbol, start_ms, end_ms, api_key, api_secret, retries=3):
    """Get OHLC data from Bybit using authenticated API"""
    url = "https://api.bybit.com/v5/market/kline"
    formatted_symbol = format_symbol_for_api(symbol)
    
    params = {
        "category": "linear",
        "symbol": formatted_symbol,
        "interval": "60",  # 1 hour
        "start": int(start_ms),
        "end": int(end_ms),
        "limit": 200
    }
    
    for attempt in range(retries):
        try:
            timestamp = str(int(time.time() * 1000))
            param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            signature_payload = timestamp + api_key + "5000" + param_str
            signature = hmac.new(api_secret.encode(), signature_payload.encode(), 'sha256').hexdigest()
            
            headers = {
                "X-BAPI-API-KEY": api_key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": "5000"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            if data.get('retCode', -1) != 0:
                logger.warning(f"Ошибка для {symbol} (Auth API): {data.get('retMsg', 'Unknown error')}")
                if attempt < retries - 1:
                    logger.debug(f"Попытка {attempt + 2}/{retries}...")
                    time.sleep(1)
                    continue
                return pd.DataFrame()
            
            # Parse data
            df = pd.DataFrame(data['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            df = df.sort_values('timestamp')
            
            logger.info(f"Получено {len(df)} свечей для {symbol} (Auth API)")
            return df
            
        except Exception as e:
            logger.warning(f"Исключение для {symbol} (Auth API): {e}")
            if attempt < retries - 1:
                logger.debug(f"Попытка {attempt + 2}/{retries}...")
                time.sleep(1)
            else:
                return pd.DataFrame()

def fetch_all_ohlc_data(symbols, start_dt, end_dt, use_auth=False, api_key=None, api_secret=None):
    """Fetch OHLC data for all symbols, with fallback to alternative pairs if needed"""
    ohlc_cache = {}
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    # ДИАГНОСТИКА: подробная информация о запросе
    logger.info(f"ДИАГНОСТИКА: Запрашиваем данные OHLC")
    logger.info(f"ДИАГНОСТИКА: Всего уникальных пар: {len(symbols)}")
    logger.info(f"ДИАГНОСТИКА: Период: c {start_dt.strftime('%Y-%m-%d %H:%M')} по {end_dt.strftime('%Y-%m-%d %H:%M')}")
    
    # Логирование всех обрабатываемых пар
    logger.info(f"Обработка {len(symbols)} уникальных пар:")
    # Вывод полного списка пар для диагностики, группами по 5
    for i in range(0, len(symbols), 5):
        batch = symbols[i:i+5]
        logger.info(f"  Пары {i+1}-{i+len(batch)}: {', '.join(batch)}")
    
    # Список символов с ошибками
    symbols_with_errors = []
    symbols_with_fallback = {}
    
    # Простая индикация прогресса
    total_symbols = len(symbols)
    logger.info(f"Загрузка OHLC данных для {total_symbols} пар...")
    
    # ДИАГНОСТИКА: проверка запроса к API
    if len(symbols) > 0:
        test_symbol = symbols[0]
        logger.info(f"ДИАГНОСТИКА: Тестовый запрос для {test_symbol}")
        logger.info(f"ДИАГНОСТИКА: URL запроса: https://api.bybit.com/v5/market/kline")
        logger.info(f"ДИАГНОСТИКА: Параметры: category=linear, symbol={format_symbol_for_api(test_symbol)}, interval=60, start={start_ms}, end={end_ms}")
    
    for i, symbol in enumerate(symbols):
        # Отображаем прогресс каждые 5 пар или в конце
        if i % 5 == 0 or i == total_symbols - 1:
            progress = (i + 1) / total_symbols * 100
            logger.info(f"Прогресс: {progress:.1f}% ({i + 1}/{total_symbols} пар)")
        
        logger.info(f"Обработка пары: {symbol} ({i+1}/{total_symbols})")
        
        # Try public API first
        if not use_auth:
            logger.debug(f"Запрос публичного API для {symbol}")
            ohlc_df = get_bybit_kline(symbol, start_ms, end_ms)
        else:
            logger.debug(f"Запрос авторизованного API для {symbol}")
            ohlc_df = get_bybit_kline_auth(symbol, start_ms, end_ms, api_key, api_secret)
        
        # If no data, try fallback pair
        if ohlc_df.empty:
            symbols_with_errors.append(symbol)
            
            # Выбираем резервную пару из расширенного словаря или значение по умолчанию
            fallback = FALLBACK_PAIRS.get(symbol, FALLBACK_PAIRS['DEFAULT'])
            logger.warning(f"Нет данных для {symbol}, используем {fallback}")
            
            if not use_auth:
                fallback_df = get_bybit_kline(fallback, start_ms, end_ms)
            else:
                fallback_df = get_bybit_kline_auth(fallback, start_ms, end_ms, api_key, api_secret)
            
            # If fallback worked, use its data
            if not fallback_df.empty:
                ohlc_cache[symbol] = fallback_df
                symbols_with_fallback[symbol] = fallback
                logger.info(f"Использованы данные {fallback} для {symbol}")
            else:
                ohlc_cache[symbol] = pd.DataFrame()
                logger.warning(f"Не удалось получить данные для {symbol} даже с использованием {fallback}")
        else:
            logger.info(f"Получены данные для {symbol}: {len(ohlc_df)} свечей")
            ohlc_cache[symbol] = ohlc_df
    
    # Вывести итоговую информацию
    if symbols_with_errors:
        logger.warning(f"Нет данных для {len(symbols_with_errors)} пар: {', '.join(symbols_with_errors[:10])}" + 
                      (f"... и еще {len(symbols_with_errors)-10} пар" if len(symbols_with_errors) > 10 else ""))
        
        # ДИАГНОСТИКА: вывести полный список пар с ошибками
        for i in range(0, len(symbols_with_errors), 5):
            batch = symbols_with_errors[i:i+5]
            logger.debug(f"  Пары с ошибками {i+1}-{i+len(batch)}: {', '.join(batch)}")
    
    if symbols_with_fallback:
        logger.info(f"Использованы резервные пары для {len(symbols_with_fallback)} символов")
        # Вывести примеры использования резервных пар
        examples = list(symbols_with_fallback.items())[:10]
        for original, fallback in examples:
            logger.info(f"  - {original} -> {fallback}")
        if len(symbols_with_fallback) > 10:
            logger.info(f"  - ... и еще {len(symbols_with_fallback) - 10} пар")
    
    # ДИАГНОСТИКА: сводка по результатам
    pairs_with_data = sum(1 for df in ohlc_cache.values() if not df.empty)
    logger.info(f"ДИАГНОСТИКА: Итог загрузки OHLC данных:")
    logger.info(f"ДИАГНОСТИКА: - Всего запрошено пар: {total_symbols}")
    logger.info(f"ДИАГНОСТИКА: - Получены данные для: {pairs_with_data} пар")
    logger.info(f"ДИАГНОСТИКА: - Нет данных для: {total_symbols - pairs_with_data} пар")
    
    return ohlc_cache

def calculate_win(signal, ohlc_df):
    """Calculate if signal hit TP or SL first and time to hit"""
    if ohlc_df.empty:
        return None, None
    
    entry_time = signal['datetime']
    entry_price = signal['entry']
    tp = signal['tp']
    sl = signal['sl']
    signal_type = signal['type'].lower() if isinstance(signal['type'], str) else str(signal['type']).lower()
    
    # Filter OHLC data to start from entry time
    relevant_ohlc = ohlc_df[ohlc_df['timestamp'] >= entry_time].copy()
    
    if relevant_ohlc.empty:
        return None, None
    
    # Check if TP or SL was hit
    for idx, row in relevant_ohlc.iterrows():
        if signal_type in ['long', 'l', '1', 'buy']:
            if row['high'] >= tp:
                # TP hit - рассчитаем время до TP
                time_to_target = (row['timestamp'] - entry_time).total_seconds() / 3600  # в часах
                return 1, time_to_target
            if row['low'] <= sl:
                # SL hit - рассчитаем время до SL
                time_to_target = (row['timestamp'] - entry_time).total_seconds() / 3600  # в часах
                return 0, time_to_target
        else:  # short
            if row['low'] <= tp:
                # TP hit
                time_to_target = (row['timestamp'] - entry_time).total_seconds() / 3600  # в часах
                return 1, time_to_target
            if row['high'] >= sl:
                # SL hit
                time_to_target = (row['timestamp'] - entry_time).total_seconds() / 3600  # в часах
                return 0, time_to_target
    
    # If we get here, neither TP nor SL was hit
    return None, None

def process_signals_batch(signals_batch, ohlc_cache):
    """Process a batch of signals to determine win/loss and time to target"""
    results = []
    times = []
    
    for _, signal in signals_batch.iterrows():
        symbol = signal['symbol']
        if symbol in ohlc_cache:
            win, time_to_target = calculate_win(signal, ohlc_cache[symbol])
        else:
            win, time_to_target = None, None
        results.append(win)
        times.append(time_to_target)
    
    return results, times

def calculate_all_wins(signals_df, ohlc_cache, use_parallel=True, max_workers=MAX_WORKERS):
    """Calculate win/loss for all signals, optionally in parallel"""
    if use_parallel and len(signals_df) > 10:
        logger.info(f"Запускаем параллельный расчет результатов сигналов с {max_workers} потоками")
        
        # Split signals into batches for parallel processing
        batch_size = max(1, len(signals_df) // max_workers)
        batches = [signals_df.iloc[i:i+batch_size] for i in range(0, len(signals_df), batch_size)]
        
        win_results = []
        time_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_signals_batch, batch, ohlc_cache) for batch in batches]
            
            total_futures = len(futures)
            logger.info(f"Расчет результатов для {len(signals_df)} сигналов в {total_futures} пакетах...")
            
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                batch_wins, batch_times = future.result()
                win_results.extend(batch_wins)
                time_results.extend(batch_times)
                
                # Отображаем прогресс
                completed += 1
                progress = completed / total_futures * 100
                if completed % 2 == 0 or completed == total_futures:
                    logger.info(f"Прогресс: {progress:.1f}% ({completed}/{total_futures} пакетов)")
        
        return win_results, time_results
    else:
        logger.info("Запускаем последовательный расчет результатов сигналов")
        results = []
        times = []
        
        total_signals = len(signals_df)
        logger.info(f"Расчет результатов для {total_signals} сигналов...")
        
        for i, (_, row) in enumerate(signals_df.iterrows()):
            win, time_to_target = calculate_win(row, ohlc_cache.get(row['symbol'], pd.DataFrame()))
            results.append(win)
            times.append(time_to_target)
            
            # Отображаем прогресс каждые 20 сигналов или в конце
            if i % 20 == 0 or i == total_signals - 1:
                progress = (i + 1) / total_signals * 100
                logger.info(f"Прогресс: {progress:.1f}% ({i + 1}/{total_signals} сигналов)")
        
        return results, times

def analyze_results(signals_df):
    """Analyze results and print detailed statistics"""
    # Calculate overall statistics
    closed_positions = signals_df[signals_df['win'].notnull()]
    pending_positions = signals_df[signals_df['win'].isnull()]
    
    if len(closed_positions) > 0:
        win_rate = closed_positions['win'].mean() * 100
    else:
        win_rate = 0
    
    # Среднее время до TP/SL
    avg_time_to_target = closed_positions['time_to_target'].mean() if len(closed_positions) > 0 and 'time_to_target' in closed_positions.columns else 0
    avg_time_to_tp = closed_positions.loc[closed_positions['win'] == 1, 'time_to_target'].mean() if len(closed_positions[closed_positions['win'] == 1]) > 0 else 0
    avg_time_to_sl = closed_positions.loc[closed_positions['win'] == 0, 'time_to_target'].mean() if len(closed_positions[closed_positions['win'] == 0]) > 0 else 0
    
    # Категории сигналов
    tp_signals = signals_df[signals_df['win'] == 1]
    sl_signals = signals_df[signals_df['win'] == 0]
    pending_signals = signals_df[signals_df['win'].isnull()]
    
    # Вывод общей статистики
    logger.info("\n=== Общая статистика ===")
    logger.info(f"Всего сигналов: {len(signals_df)}")
    logger.info(f"Закрытых позиций: {len(closed_positions)}")
    logger.info(f"Pending позиций: {len(pending_signals)}")
    logger.info(f"Общий винрейт: {win_rate:.2f}%")
    logger.info(f"Среднее время до TP: {avg_time_to_tp:.1f} часов")
    logger.info(f"Среднее время до SL: {avg_time_to_sl:.1f} часов")
    
    # Анализ по категориям
    logger.info("\n=== Результаты по категориям ===")
    
    # TP сигналы
    tp_count = len(tp_signals)
    tp_percent = (tp_count / len(signals_df)) * 100 if len(signals_df) > 0 else 0
    logger.info(f"\nTP сигналы (win=1): {tp_count} ({tp_percent:.1f}%)")
    if tp_count > 0:
        logger.info("Примеры TP сигналов:")
        for _, row in tp_signals.head().iterrows():
            logger.info(f"  {row['symbol']} {row['type']} вход={row['entry']:.4f} TP через {row['time_to_target']:.1f}ч")
    
    # SL сигналы
    sl_count = len(sl_signals)
    sl_percent = (sl_count / len(signals_df)) * 100 if len(signals_df) > 0 else 0
    logger.info(f"\nSL сигналы (win=0): {sl_count} ({sl_percent:.1f}%)")
    if sl_count > 0:
        logger.info("Примеры SL сигналов:")
        for _, row in sl_signals.head().iterrows():
            logger.info(f"  {row['symbol']} {row['type']} вход={row['entry']:.4f} SL через {row['time_to_target']:.1f}ч")
    
    # Pending сигналы
    pending_count = len(pending_signals)
    pending_percent = (pending_count / len(signals_df)) * 100 if len(signals_df) > 0 else 0
    logger.info(f"\nPending сигналы: {pending_count} ({pending_percent:.1f}%)")
    if pending_count > 0:
        logger.info("Примеры Pending сигналов:")
        for _, row in pending_signals.head().iterrows():
            logger.info(f"  {row['symbol']} {row['type']} вход={row['entry']:.4f}")
    
    # Анализ по парам
    logger.info("\n=== Статистика по парам ===")
    for symbol in signals_df['symbol'].unique():
        symbol_df = signals_df[signals_df['symbol'] == symbol]
        closed_symbol = symbol_df[symbol_df['win'].notnull()]
        
        if len(closed_symbol) > 0:
            symbol_win_rate = closed_symbol['win'].mean() * 100
            symbol_avg_time = closed_symbol['time_to_target'].mean()
            logger.info(f"{symbol}: {len(symbol_df)} сигналов, {symbol_win_rate:.1f}% винрейт, среднее время {symbol_avg_time:.1f}ч")
    
    # Анализ по типам сигналов
    logger.info("\n=== Статистика по типам сигналов ===")
    for signal_type in ['long', 'short']:
        type_df = closed_positions[closed_positions['type'] == signal_type]
        if len(type_df) > 0:
            type_win_rate = type_df['win'].mean() * 100
            type_avg_time = type_df['time_to_target'].mean()
            logger.info(f"{signal_type.upper()}: {len(type_df)} сигналов, {type_win_rate:.1f}% винрейт, среднее время {type_avg_time:.1f}ч")
    
    # Возвращаем статистику
    return {
        'win_rate': win_rate,
        'closed_positions': len(closed_positions),
        'pending_positions': len(pending_signals),
        'avg_time_to_tp': avg_time_to_tp,
        'avg_time_to_sl': avg_time_to_sl,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'pending_count': pending_count
    }

def create_test_data(test_df=None, pairs_limit=None):
    """Создает тестовые данные, если файл не найден или некорректен"""
    logger.info("==== НАЧАЛО СОЗДАНИЯ ТЕСТОВЫХ ДАННЫХ ====")
    logger.info(f"ДИАГНОСТИКА: Значение параметра pairs_limit = {pairs_limit}")
    
    if test_df is not None:
        logger.info(f"ДИАГНОСТИКА: Получен DataFrame с {len(test_df)} строками")
        if 'symbol' in test_df.columns:
            logger.info(f"ДИАГНОСТИКА: В переданном DataFrame {len(test_df['symbol'].unique())} уникальных пар")
    
    # Проверяем, есть ли ограничение на количество пар
    num_pairs = 74 if pairs_limit is None else pairs_limit
    logger.info(f"ОПЕРАЦИЯ: Установлено ограничение на {num_pairs} пар")
    
    # Список криптовалютных пар
    pairs = [
        # Основные пары
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "SOLUSDT", "DOTUSDT", "TRXUSDT", "AVAXUSDT",
        "LINKUSDT", "MATICUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT",
        # Дополнительные пары до 74 штук
        "BCHUSDT", "ETCUSDT", "XLMUSDT", "VETUSDT", "THETAUSDT",
        "FILUSDT", "XMRUSDT", "ICPUSDT", "AAVEUSDT", "EGLDUSDT",
        "AXSUSDT", "XTZUSDT", "EOSUSDT", "CAKEUSDT", "ALGOUSDT",
        "NEOUSDT", "FTMUSDT", "WAVESUSDT", "KSMUSDT", "DASHUSDT",
        "MKRUSDT", "RUNEUSDT", "NEARUSDT", "QNTUSDT", "FTTUSDT",
        "HBARUSDT", "ZECUSDT", "ENJUSDT", "CHZUSDT", "MANAUSDT",
        "HOTUSDT", "SANDUSDT", "FLOWUSDT", "GALAUSDT", "TFUELUSDT",
        "SHIBUSDT", "GRTUSDT", "BATUSDT", "ONEUSDT", "ZILUSDT",
        "SUSHIUSDT", "CELOUSDT", "CRVUSDT", "HNTUSDT", "ZRXUSDT",
        "IOTXUSDT", "LRCUSDT", "RENUSDT", "IOTAUSDT", "SRMUSDT",
        "BTCBUSD", "ETHBUSD", "BNBBUSD", "ADABUSD", "XRPBUSD",
        "DOGEBUSD", "SOLBUSD", "DOTBUSD", "MATICBUSD", "LINKBUSD",
        "UNIBUSD", "AAVEBUSD", "SXPBUSD", "LUNABUSD"
    ]
    
    logger.info(f"ОПЕРАЦИЯ: Загружен исходный список из {len(pairs)} пар")
    
    # Если передан pairs_limit, ограничиваем количество пар
    if pairs_limit is not None and pairs_limit < len(pairs):
        logger.warning(f"ОПЕРАЦИЯ: Ограничиваем количество пар до {pairs_limit}")
        pairs = pairs[:pairs_limit]
    
    logger.info(f"ОПЕРАЦИЯ: После применения ограничений осталось {len(pairs)} пар")
    
    # Форматируем пары для совместимости
    logger.info("ОПЕРАЦИЯ: Форматирование символов пар")
    formatted_pairs = [pair.replace('USDT', '/USDT').replace('BUSD', '/BUSD') for pair in pairs]
    logger.info(f"ОПЕРАЦИЯ: Пары отформатированы: {', '.join(formatted_pairs[:10])}... (всего {len(formatted_pairs)})")
    
    # Создаем тестовые данные
    logger.info("ОПЕРАЦИЯ: Начинаем генерацию случайных сигналов для пар")
    signals = []
    current_datetime = datetime.now()
    
    # Количество сигналов для каждой пары
    signals_per_pair = 5
    logger.info(f"ОПЕРАЦИЯ: Создаем {signals_per_pair} сигналов для каждой пары")
    
    total_signals_count = 0
    
    # Создаем сигналы для каждой пары
    logger.info("ОПЕРАЦИЯ: Генерация сигналов для каждой пары")
    for pair in formatted_pairs:
        logger.info(f"ОПЕРАЦИЯ: Генерация сигналов для пары {pair}")
        for i in range(signals_per_pair):
            # Генерируем случайные данные
            is_long = random.choice([True, False])
            signal_type = 'long' if is_long else 'short'
            
            # Генерируем случайные цены
            entry_price = random.uniform(100, 50000)
            
            # TP и SL зависят от типа сигнала
            if is_long:
                tp_price = entry_price * (1 + random.uniform(0.01, 0.1))
                sl_price = entry_price * (1 - random.uniform(0.01, 0.05))
            else:
                tp_price = entry_price * (1 - random.uniform(0.01, 0.1))
                sl_price = entry_price * (1 + random.uniform(0.01, 0.05))
            
            # Создаем временную метку со смещением
            signal_datetime = current_datetime - timedelta(hours=i*4)
            
            signals.append({
                'symbol': pair,
                'type': signal_type,
                'entry': entry_price,
                'tp': tp_price,
                'sl': sl_price,
                'datetime': signal_datetime.strftime('%Y-%m-%d %H:%M:%S')
            })
            total_signals_count += 1
    
    logger.info(f"ОПЕРАЦИЯ: Завершена генерация всех сигналов")
    logger.info(f"ОПЕРАЦИЯ: Создано {total_signals_count} сигналов для {len(formatted_pairs)} пар")
    
    # Создаем DataFrame из сигналов
    logger.info("ОПЕРАЦИЯ: Создание DataFrame из сгенерированных сигналов")
    result_df = pd.DataFrame(signals)
    
    logger.info(f"ОПЕРАЦИЯ: Создан DataFrame с {len(result_df)} строками и {len(result_df['symbol'].unique())} уникальными парами")
    logger.info(f"ОПЕРАЦИЯ: Уникальные пары: {', '.join(sorted(result_df['symbol'].unique())[:10])}... (и другие)")
    logger.info("==== ЗАВЕРШЕНИЕ СОЗДАНИЯ ТЕСТОВЫХ ДАННЫХ ====")
    
    return result_df

def read_data(file_path):
    """Чтение данных из Excel файла."""
    logger.info("==== НАЧАЛО ЧТЕНИЯ ДАННЫХ ====")
    logger.info(f"ОПЕРАЦИЯ: Чтение данных из файла: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"ОПЕРАЦИЯ: Файл не найден: {file_path}")
        return None
    
    try:
        # Определяем расширение файла
        _, ext = os.path.splitext(file_path)
        logger.info(f"ОПЕРАЦИЯ: Определение типа файла: {ext}")
        
        if ext.lower() == '.csv':
            logger.info(f"ОПЕРАЦИЯ: Чтение CSV файла {file_path}")
            df = pd.read_csv(file_path)
        elif ext.lower() in ['.xls', '.xlsx']:
            logger.info(f"ОПЕРАЦИЯ: Чтение Excel файла {file_path}")
            df = pd.read_excel(file_path)
        else:
            logger.error(f"ОПЕРАЦИЯ: Неподдерживаемый формат файла: {ext}")
            return None
        
        logger.info(f"ОПЕРАЦИЯ: Считано {len(df)} строк из файла")
        
        # Проверяем необходимые колонки
        required_columns = ['symbol', 'type', 'entry', 'tp', 'sl', 'datetime']
        logger.info(f"ОПЕРАЦИЯ: Проверка наличия необходимых колонок: {', '.join(required_columns)}")
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"ОПЕРАЦИЯ: В файле отсутствуют необходимые колонки: {', '.join(missing_columns)}")
            logger.error(f"ОПЕРАЦИЯ: Доступные колонки: {', '.join(df.columns)}")
            return None
            
        # Преобразование типов данных
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df[['entry', 'tp', 'sl']] = df[['entry', 'tp', 'sl']].astype(float)
            df['type'] = df['type'].str.lower()
        except Exception as e:
            logger.error(f"ОПЕРАЦИЯ: Ошибка преобразования типов данных: {e}")
            return None
        
        # Диагностика данных
        if 'symbol' in df.columns:
            unique_pairs = df['symbol'].unique()
            logger.info(f"ОПЕРАЦИЯ: Найдено {len(unique_pairs)} уникальных пар в файле")
            logger.info(f"ОПЕРАЦИЯ: Примеры пар: {', '.join(sorted(unique_pairs)[:10])}")
            
            # Подсчет количества сигналов для каждой пары
            signals_by_pair = df['symbol'].value_counts()
            logger.info(f"ОПЕРАЦИЯ: Статистика по количеству сигналов на пару:")
            logger.info(f"  - Минимальное: {signals_by_pair.min()}")
            logger.info(f"  - Максимальное: {signals_by_pair.max()}")
            logger.info(f"  - Среднее: {signals_by_pair.mean():.1f}")
            
            # Вывести топ-10 пар по количеству сигналов
            top_pairs = signals_by_pair.nlargest(10)
            logger.info(f"ОПЕРАЦИЯ: Топ-10 пар по количеству сигналов:")
            for symbol, count in top_pairs.items():
                logger.info(f"  - {symbol}: {count} сигналов")
        
        if 'type' in df.columns:
            # Нормализуем тип сигнала к нижнему регистру
            logger.info("ОПЕРАЦИЯ: Нормализация типов сигналов к нижнему регистру")
            df['type'] = df['type'].str.lower()
            type_counts = df['type'].value_counts()
            logger.info(f"ОПЕРАЦИЯ: Количество long сигналов: {type_counts.get('long', 0)}")
            logger.info(f"ОПЕРАЦИЯ: Количество short сигналов: {type_counts.get('short', 0)}")
        
        # Проверка на пустые значения
        na_count = df[required_columns].isna().sum()
        if na_count.sum() > 0:
            logger.info("ОПЕРАЦИЯ: Проверка на пустые значения в обязательных колонках")
            for col in required_columns:
                if na_count[col] > 0:
                    logger.warning(f"ОПЕРАЦИЯ: Колонка '{col}' содержит {na_count[col]} пустых значений")
        
        logger.info("==== ЗАВЕРШЕНИЕ ЧТЕНИЯ ДАННЫХ ====")
        return df
    
    except Exception as e:
        logger.error(f"ОПЕРАЦИЯ: Ошибка при чтении файла: {e}")
        logger.exception("Подробная информация об ошибке:")
        return None

def main():
    """Основная функция для анализа торговых сигналов"""
    try:
        logger.info("==== НАЧАЛО ВЫПОЛНЕНИЯ ПРОГРАММЫ ====")
        
        # Проверяем наличие входного файла
        logger.info(f"ОПЕРАЦИЯ: Проверка наличия файла: {INPUT_FILE}")
        if not os.path.exists(INPUT_FILE):
            logger.error(f"Ошибка: Файл {INPUT_FILE} не найден!")
            return

        # Чтение данных из Excel
        logger.info(f"ОПЕРАЦИЯ: Начинаем чтение данных из Excel файла {INPUT_FILE}")
        signals_df = read_data(INPUT_FILE)
        
        if signals_df is None or signals_df.empty:
            logger.error("ОПЕРАЦИЯ: Не удалось прочитать данные из файла или данные пусты")
            return
        
        # Выводим информацию о загруженных данных
        logger.info(f"Загружено {len(signals_df)} строк из файла {INPUT_FILE}")
        min_date = signals_df['datetime'].min()
        max_date = signals_df['datetime'].max()
        logger.info(f"Временной диапазон: с {min_date.strftime('%Y-%m-%d %H:%M:%S')} по {max_date.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Уникальных пар: {len(signals_df['symbol'].unique())}")
        
        # Фильтрация аномалий
        logger.info("ОПЕРАЦИЯ: Начинаем фильтрацию аномалий в данных")
        filtered_df = filter_anomalies(signals_df)
        
        if filtered_df is None or filtered_df.empty:
            logger.error("ОПЕРАЦИЯ: После фильтрации аномалий не осталось данных")
            return
        
        # Определение временного диапазона для данных OHLC
        min_datetime = filtered_df['datetime'].min()
        max_datetime = filtered_df['datetime'].max() + pd.Timedelta(hours=24)  # +24 часа для гарантии захвата всех событий
        
        logger.info(f"ОПЕРАЦИЯ: Временной диапазон для OHLC: {min_datetime.strftime('%Y-%m-%d %H:%M:%S')} - {max_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            
        # Получение списка уникальных пар
        symbols = filtered_df['symbol'].unique().tolist()
        logger.info(f"ОПЕРАЦИЯ: Обнаружено {len(symbols)} уникальных пар")
        
        # Получение OHLC данных для всех пар
        logger.info("ОПЕРАЦИЯ: Получение OHLC данных для всех пар")
        ohlc_cache = fetch_all_ohlc_data(symbols, min_datetime, max_datetime)
        
        # Расчет win и time_to_target для всех сигналов
        logger.info("ОПЕРАЦИЯ: Расчет win и time_to_target для всех сигналов")
        win_results, time_results = calculate_all_wins(filtered_df, ohlc_cache, use_parallel=True, max_workers=MAX_WORKERS)
        
        # Добавление результатов в DataFrame
        filtered_df['win'] = win_results
        filtered_df['time_to_target'] = time_results
        
        # Анализ результатов
        logger.info("ОПЕРАЦИЯ: Анализ результатов")
        analysis_results = analyze_results(filtered_df)
        
        # Переименовываем datetime в timestamp
        filtered_df.rename(columns={'datetime': 'timestamp'}, inplace=True)
        
        # Сохранение результатов в Excel
        logger.info(f"ОПЕРАЦИЯ: Сохранение результатов в файл {OUTPUT_FILE}")
        filtered_df.to_excel(OUTPUT_FILE, index=False)
        logger.info(f"ОПЕРАЦИЯ: Результаты сохранены в файл {OUTPUT_FILE}")
        
        logger.info("==== ЗАВЕРШЕНИЕ ВЫПОЛНЕНИЯ ПРОГРАММЫ ====")

    except Exception as e:
        logger.error(f"Ошибка при выполнении функции main(): {e}")
        logger.exception("Подробная информация об ошибке:")

if __name__ == "__main__":
    main() 