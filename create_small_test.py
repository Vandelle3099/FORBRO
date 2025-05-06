import analyze

# Модифицируем функцию main для ограничения количества пар
def test_main():
    try:
        # Загружаем данные из Excel
        import pandas as pd
        
        # Создаем малый тестовый набор данных
        test_df = pd.DataFrame({
            'symbol': ['BTC/USDT', 'ETH/USDT', 'DOGE/USDT', 'SOL/USDT'],
            'type': ['long', 'short', 'long', 'short'],
            'entry': [50000, 3000, 0.15, 150],
            'tp': [55000, 2700, 0.17, 135],
            'sl': [48000, 3200, 0.14, 165],
            'datetime': ['2025-04-25 12:00:00', '2025-04-25 13:00:00', '2025-04-25 14:00:00', '2025-04-25 15:00:00']
        })
        
        # Запускаем создание тестовых данных с ограничением на 5 пар
        test_data = analyze.create_test_data(test_df, pairs_limit=5)
        
        # Фильтруем аномалии
        filtered_df = analyze.filter_anomalies(test_data)
        
        print(f"Успешно создано и отфильтровано {len(filtered_df)} строк с {len(filtered_df['symbol'].unique())} уникальными парами")
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    test_main() 