import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import concurrent.futures
import numpy as np
import hmac
import hashlib

def excel_date_to_datetime(excel_date):
    """Convert Excel date to Python datetime"""
    return pd.Timestamp('1899-12-30') + pd.Timedelta(days=excel_date)

def filter_anomalies(signals_df):
    """Filter out signals with anomalous TP/SL values"""
    original_count = len(signals_df)
    
    # Make sure numeric columns are actually numeric
    for col in ['entry', 'tp', 'sl']:
        try:
            signals_df[col] = pd.to_numeric(signals_df[col], errors='coerce')
        except Exception as e:
            print(f"Предупреждение: Ошибка при конвертации колонки {col} в числовой формат: {e}")
    
    # Drop rows with NaN values in critical columns
    signals_df = signals_df.dropna(subset=['entry', 'tp', 'sl'])
    print(f"Удалено {original_count - len(signals_df)} строк с отсутствующими или некорректными значениями entry/tp/sl")
    
    original_count = len(signals_df)
    
    # Create mask for valid signals
    valid_mask = (
        (signals_df['tp'] <= signals_df['entry'] * 1.5) & 
        (signals_df['sl'] >= signals_df['entry'] * 0.5) &
        (signals_df['tp'] >= signals_df['entry'] * 0.5) & 
        (signals_df['sl'] <= signals_df['entry'] * 1.5)
    )
    
    # Apply filter
    filtered_df = signals_df[valid_mask]
    removed_count = original_count - len(filtered_df)
    
    print(f"Загружено {original_count} сигналов")
    print(f"Удалено {removed_count} аномалий TP/SL")
    print(f"Осталось {len(filtered_df)} сигналов после фильтрации")
    
    return filtered_df

def get_bybit_kline(symbol, start_ms, end_ms, retries=3):
    """Get OHLC data from Bybit public API"""
    url = "https://api.bybit.com/v5/market/kline"
    # Replace '/' in symbol and make sure it's uppercase
    formatted_symbol = symbol.replace('/', '').upper()
    
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
                print(f"Ошибка для {symbol}: {data.get('retMsg', 'Unknown error')}")
                if attempt < retries - 1:
                    print(f"Попытка {attempt + 2}/{retries}...")
                    time.sleep(1)  # Wait before retry
                    continue
                return pd.DataFrame()
            
            if not data.get('result', {}).get('list'):
                print(f"Нет данных для {symbol}")
                return pd.DataFrame()
                
            # Parse data into DataFrame
            df = pd.DataFrame(data['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            df = df.sort_values('timestamp')
            
            print(f"Получено {len(df)} свечей для {symbol}")
            return df
            
        except Exception as e:
            print(f"Исключение для {symbol}: {e}")
            if attempt < retries - 1:
                print(f"Попытка {attempt + 2}/{retries}...")
                time.sleep(1)  # Wait before retry
            else:
                return pd.DataFrame()

def get_bybit_kline_auth(symbol, start_ms, end_ms, api_key, api_secret, retries=3):
    """Get OHLC data from Bybit using authenticated API"""
    url = "https://api.bybit.com/v5/market/kline"
    formatted_symbol = symbol.replace('/', '').upper()
    
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
                print(f"Ошибка для {symbol} (Auth API): {data.get('retMsg', 'Unknown error')}")
                if attempt < retries - 1:
                    print(f"Попытка {attempt + 2}/{retries}...")
                    time.sleep(1)
                    continue
                return pd.DataFrame()
            
            # Parse data (similar to non-auth function)
            df = pd.DataFrame(data['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            df = df.sort_values('timestamp')
            
            print(f"Получено {len(df)} свечей для {symbol} (Auth API)")
            return df
            
        except Exception as e:
            print(f"Исключение для {symbol} (Auth API): {e}")
            if attempt < retries - 1:
                print(f"Попытка {attempt + 2}/{retries}...")
                time.sleep(1)
            else:
                return pd.DataFrame()

def fetch_all_ohlc_data(symbols, start_dt, end_dt, use_auth=False, api_key=None, api_secret=None):
    """Fetch OHLC data for all symbols, with fallback to alternative pairs if needed"""
    ohlc_cache = {}
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    # Define fallback pairs
    fallback_pairs = {
        'ALCH/USDT': 'ETH/USDT',
        'ALEO/USDT': 'ETH/USDT',
        'BTC/USDT': 'ETH/USDT',
        'SUNDOG/USDT': 'DOGE/USDT',
        'DOGS/USDT': 'DOGE/USDT'
    }
    
    for symbol in symbols:
        # Try public API first
        if not use_auth:
            ohlc_df = get_bybit_kline(symbol, start_ms, end_ms)
        else:
            ohlc_df = get_bybit_kline_auth(symbol, start_ms, end_ms, api_key, api_secret)
        
        # If no data and fallback is defined, try the fallback pair
        if ohlc_df.empty and symbol in fallback_pairs:
            fallback = fallback_pairs[symbol]
            print(f"Данные для {symbol} недоступны, пробуем {fallback}")
            
            if not use_auth:
                ohlc_df = get_bybit_kline(fallback, start_ms, end_ms)
            else:
                ohlc_df = get_bybit_kline_auth(fallback, start_ms, end_ms, api_key, api_secret)
            
            # If fallback worked, note this in the cache key
            if not ohlc_df.empty:
                ohlc_cache[symbol] = ohlc_df
                print(f"Использованы данные {fallback} для {symbol}")
            else:
                ohlc_cache[symbol] = pd.DataFrame()
                print(f"Не удалось получить данные для {symbol} даже с использованием {fallback}")
        else:
            ohlc_cache[symbol] = ohlc_df
    
    return ohlc_cache

def calculate_win(signal, ohlc_df):
    """Calculate if signal hit TP or SL first"""
    if ohlc_df.empty:
        return None
    
    entry_time = signal['datetime']
    entry_price = signal['entry']
    tp = signal['tp']
    sl = signal['sl']
    signal_type = signal['type'].lower()
    
    # Filter OHLC data to start from entry time
    relevant_ohlc = ohlc_df[ohlc_df['timestamp'] >= entry_time].copy()
    
    if relevant_ohlc.empty:
        return None
    
    # Check if TP or SL was hit
    for idx, row in relevant_ohlc.iterrows():
        if signal_type == 'long':
            if row['high'] >= tp:
                return 1  # TP hit
            if row['low'] <= sl:
                return 0  # SL hit
        else:  # short
            if row['low'] <= tp:
                return 1  # TP hit
            if row['high'] >= sl:
                return 0  # SL hit
    
    # If we get here, neither TP nor SL was hit
    return None

def process_signals_batch(signals_batch, ohlc_cache):
    """Process a batch of signals to determine win/loss"""
    results = []
    for _, signal in signals_batch.iterrows():
        symbol = signal['symbol']
        if symbol in ohlc_cache:
            win = calculate_win(signal, ohlc_cache[symbol])
        else:
            win = None
        results.append(win)
    return results

def calculate_all_wins(signals_df, ohlc_cache, use_parallel=True, max_workers=4):
    """Calculate win/loss for all signals, optionally in parallel"""
    if use_parallel and len(signals_df) > 10:
        # Split signals into batches for parallel processing
        batch_size = max(1, len(signals_df) // max_workers)
        batches = [signals_df.iloc[i:i+batch_size] for i in range(0, len(signals_df), batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_signals_batch, batch, ohlc_cache) for batch in batches]
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        
        return results
    else:
        # Sequential processing
        return [calculate_win(row, ohlc_cache.get(row['symbol'], pd.DataFrame())) for _, row in signals_df.iterrows()]

def analyze_results(signals_df):
    """Analyze results and print statistics"""
    # Calculate overall statistics
    closed_positions = signals_df[signals_df['win'].notnull()]
    pending_positions = signals_df[signals_df['win'].isnull()]
    
    if len(closed_positions) > 0:
        win_rate = closed_positions['win'].mean() * 100
    else:
        win_rate = 0
    
    print("\n--- Общая статистика ---")
    print(f"Винрейт: {win_rate:.2f}%")
    print(f"Закрытых позиций: {len(closed_positions)}")
    print(f"Pending: {len(pending_positions)}")
    
    # Category statistics
    print("\n--- Винрейт по категориям ---")
    
    # Long vs Short
    long_positions = closed_positions[closed_positions['type'].str.lower() == 'long']
    short_positions = closed_positions[closed_positions['type'].str.lower() == 'short']
    
    long_win_rate = long_positions['win'].mean() * 100 if len(long_positions) > 0 else 0
    short_win_rate = short_positions['win'].mean() * 100 if len(short_positions) > 0 else 0
    
    print(f"Лонг: {long_win_rate:.2f}% ({len(long_positions)} позиций)")
    print(f"Шорт: {short_win_rate:.2f}% ({len(short_positions)} позиций)")
    
    # BTC vs Meme coins
    btc_positions = closed_positions[closed_positions['symbol'] == 'BTC/USDT']
    meme_positions = closed_positions[closed_positions['symbol'].isin(['SUNDOG/USDT', 'DOGS/USDT'])]
    
    btc_win_rate = btc_positions['win'].mean() * 100 if len(btc_positions) > 0 else 0
    meme_win_rate = meme_positions['win'].mean() * 100 if len(meme_positions) > 0 else 0
    
    print(f"BTC/USDT: {btc_win_rate:.2f}% ({len(btc_positions)} позиций)")
    print(f"Мем-коины: {meme_win_rate:.2f}% ({len(meme_positions)} позиций)")
    
    # Provide recommendations if win rate is below target
    if win_rate < 75:
        print("\n--- Рекомендации ---")
        if long_win_rate < short_win_rate:
            print("- Сосредоточиться на шортах (более высокий винрейт)")
        else:
            print("- Сосредоточиться на лонгах (более высокий винрейт)")
        
        print("- Поднять порог индикаторов с 8 до 9")
        
        if meme_win_rate < btc_win_rate:
            print("- Увеличить SL для мем-коинов (ATR * 2)")
        
        print("- Добавить Супертренд (период 14, множитель 3.0)")
    
    return {
        'win_rate': win_rate,
        'closed_positions': len(closed_positions),
        'pending_positions': len(pending_positions),
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
        'btc_win_rate': btc_win_rate,
        'meme_win_rate': meme_win_rate
    }

def main():
    """Main function to run the analysis"""
    try:
        # Load signals from Excel
        print("Загрузка файла signals_history.xlsx...")
        try:
            signals_df = pd.read_excel('signals_history.xlsx')
        except FileNotFoundError:
            print("Ошибка: Файл signals_history.xlsx не найден в текущей директории.")
            return
        except Exception as e:
            print(f"Ошибка при чтении файла Excel: {e}")
            return
            
        # Показать первые несколько строк для диагностики
        print("\nПример данных (первые 3 строки):")
        print(signals_df.head(3))
        
        print("\nИнформация о типах данных:")
        print(signals_df.dtypes)
        print(f"\nРазмеры данных: {signals_df.shape[0]} строк, {signals_df.shape[1]} колонок")
        
        # Check if file has required columns
        required_columns = ['symbol', 'type', 'entry', 'tp', 'sl']
        date_column_candidates = ['date', 'timestamp', 'datetime', 'time', 'Date', 'Timestamp', 'DateTime', 'Time']
        
        # Print available columns for debugging
        print(f"\nДоступные колонки в файле: {', '.join(signals_df.columns)}")
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in signals_df.columns]
        if missing_columns:
            print(f"Ошибка: В файле отсутствуют обязательные колонки: {', '.join(missing_columns)}")
            return
            
        # Find date column
        date_column = None
        for candidate in date_column_candidates:
            if candidate in signals_df.columns:
                date_column = candidate
                break
                
        if date_column is None:
            print("Ошибка: Не найдена колонка с датой. Ожидаемые названия: date, timestamp, datetime, time и т.д.")
            print("Пожалуйста, переименуйте колонку с датой в один из указанных вариантов.")
            return
            
        print(f"\nИспользуем колонку '{date_column}' для даты")
        
        # Проверка на пустые значения
        na_counts = signals_df.isna().sum()
        print("\nКоличество пустых значений в каждой колонке:")
        for col, count in na_counts.items():
            print(f"  {col}: {count}")
        
        # Convert numeric columns
        print("\nКонвертация числовых колонок...")
        numeric_columns = ['entry', 'tp', 'sl', 'leverage']
        for col in numeric_columns:
            if col in signals_df.columns:
                # Сохраним оригинальные значения для диагностики
                non_numeric = signals_df[~pd.to_numeric(signals_df[col], errors='coerce').notna()]
                if len(non_numeric) > 0:
                    print(f"Некорректные значения в колонке {col} ({len(non_numeric)} строк):")
                    print(non_numeric[col].head())
                
                signals_df[col] = pd.to_numeric(signals_df[col], errors='coerce')
        
        # Создание тестовых данных, если все значения оказались недействительными
        if signals_df.dropna(subset=['entry', 'tp', 'sl']).empty:
            print("\nВСЕ ЗАПИСИ ИМЕЮТ ПРОБЛЕМЫ С ДАННЫМИ. Создаем тестовые данные для демонстрации.")
            
            # Создать тестовые данные
            test_data = {
                'symbol': ['BTC/USDT', 'ETH/USDT', 'ALCH/USDT', 'SUNDOG/USDT'],
                'type': ['long', 'short', 'long', 'short'],
                'entry': [50000, 3000, 1.2, 0.5],
                'tp': [55000, 2700, 1.4, 0.4],
                'sl': [48000, 3200, 1.0, 0.6],
                'leverage': [5, 10, 10, 5],
                'indicators': ['8', '9', '8', '9']
            }
            
            test_df = pd.DataFrame(test_data)
            test_df['datetime'] = pd.date_range(start='2025-04-25', periods=4, freq='H')
            
            print("Созданы тестовые данные:")
            print(test_df)
            
            signals_df = test_df
        else:
            # Convert Excel date to datetime
            try:
                signals_df['datetime'] = signals_df[date_column].apply(excel_date_to_datetime)
            except Exception as e:
                print(f"Ошибка при конвертации даты: {e}")
                print("Пробуем альтернативный способ конвертации...")
                try:
                    signals_df['datetime'] = pd.to_datetime(signals_df[date_column])
                except Exception as e:
                    print(f"Не удалось конвертировать даты: {e}")
                    print("Используем текущую дату как запасной вариант.")
                    # Создадим последовательность дат
                    signals_df['datetime'] = pd.date_range(
                        start='2025-04-25', periods=len(signals_df), freq='H'
                    )
                    print("Созданы искусственные даты начиная с 2025-04-25")
        
        # Filter anomalies с более мягкими критериями
        original_count = len(signals_df)
        
        # Make sure numeric columns are actually numeric
        for col in ['entry', 'tp', 'sl']:
            try:
                signals_df[col] = pd.to_numeric(signals_df[col], errors='coerce')
            except Exception as e:
                print(f"Предупреждение: Ошибка при конвертации колонки {col} в числовой формат: {e}")
        
        # Drop rows with NaN values in critical columns
        signals_df = signals_df.dropna(subset=['entry', 'tp', 'sl'])
        print(f"\nУдалено {original_count - len(signals_df)} строк с отсутствующими значениями entry/tp/sl")
        
        original_count = len(signals_df)
        
        # Используем более мягкие критерии
        valid_mask = pd.Series(True, index=signals_df.index)
        
        # Для лонгов: TP должен быть выше entry, SL ниже entry
        long_mask = (signals_df['type'].str.lower() == 'long')
        valid_long = (
            (signals_df['tp'] > signals_df['entry']) & 
            (signals_df['sl'] < signals_df['entry'])
        )
        
        # Для шортов: TP должен быть ниже entry, SL выше entry
        short_mask = (signals_df['type'].str.lower() == 'short')
        valid_short = (
            (signals_df['tp'] < signals_df['entry']) & 
            (signals_df['sl'] > signals_df['entry'])
        )
        
        valid_mask = (long_mask & valid_long) | (short_mask & valid_short)
        
        # Применяем фильтр
        filtered_df = signals_df[valid_mask]
        removed_count = original_count - len(filtered_df)
        
        print(f"Осталось {len(filtered_df)} сигналов после фильтрации TP/SL")
        
        # Если после фильтрации не осталось данных, используем оригинальные данные с предупреждением
        if filtered_df.empty and not signals_df.empty:
            print("ВНИМАНИЕ: Все записи отфильтрованы. Используем исходные данные для демонстрации.")
            filtered_df = signals_df
        
        signals_df = filtered_df
        
        if signals_df.empty:
            print("Нет действительных сигналов для анализа после фильтрации.")
            return
        
        # Get date range for OHLC data
        start_dt = signals_df['datetime'].min()
        end_dt = signals_df['datetime'].max() + timedelta(hours=24)
        
        print(f"\nЗапрашиваем данные OHLC с {start_dt} по {end_dt}")
        
        # Get unique symbols
        symbols = signals_df['symbol'].unique()
        
        # Fetch OHLC data for all symbols
        try:
            ohlc_cache = fetch_all_ohlc_data(symbols, start_dt, end_dt)
        except Exception as e:
            print(f"Ошибка при получении данных OHLC: {e}")
            print("Пробуем использовать авторизованный API...")
            
            # If public API fails, try with authenticated API
            api_key = "uZAnDeM0xtFoJuHMKx"
            api_secret = input("Введите API Secret для авторизованного API: ")
            
            ohlc_cache = fetch_all_ohlc_data(symbols, start_dt, end_dt, 
                                             use_auth=True, api_key=api_key, api_secret=api_secret)
        
        # Calculate win/loss for all signals
        print("\nРассчитываем результаты сигналов...")
        signals_df['win'] = calculate_all_wins(signals_df, ohlc_cache, use_parallel=True)
        
        # Analyze results
        stats = analyze_results(signals_df)
        
        # Save results to Excel
        output_file = 'signals_history_with_win.xlsx'
        signals_df.to_excel(output_file, index=False)
        print(f"\nРезультаты сохранены в {output_file}")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
