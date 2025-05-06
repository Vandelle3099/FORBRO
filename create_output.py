import pandas as pd

# Создаем тестовый DataFrame с необходимыми колонками
df = pd.DataFrame({
    'symbol': ['BTC/USDT', 'ETH/USDT', 'DOGE/USDT', 'SOL/USDT'],
    'type': ['long', 'short', 'long', 'short'],
    'entry': [50000, 3000, 0.15, 150],
    'tp': [55000, 2700, 0.17, 135],
    'sl': [48000, 3200, 0.14, 165],
    'datetime': ['2025-04-25 12:00:00', '2025-04-25 13:00:00', '2025-04-25 14:00:00', '2025-04-25 15:00:00']
})

# Сохраняем в Excel
df.to_excel('output.xlsx', index=False)

print("Файл output.xlsx успешно создан") 