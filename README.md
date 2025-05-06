# Crypto Signals Bot

Асинхронный бот для генерации торговых сигналов на фьючерсных парах Bybit (USDT) с использованием машинного обучения, паттернов и связок технических индикаторов.

## Функциональность

- **Прогнозирование** - используется комбинация ML-моделей (RandomForest), технических индикаторов и анализа паттернов
- **Индикаторы** - 23+ технических индикатора (RSI, MACD, Ichimoku, Bollinger Bands, VWAP, ATR и др.)
- **Сигналы** - генерация сигналов для Long/Short позиций с целями TP/SL
- **Telegram** - отправка уведомлений в Telegram о сигналах и их исходах
- **Аналитика** - генерация отчетов по эффективности сигналов и ML-моделей
- **Фильтры** - анализ объемов, волатильности, новостей

## ML-функциональность

- **RandomForest** модель для прогнозирования движения цены
- **Автоматическое обучение** на исторических данных (до 240 свечей)
- **Регулярное переобучение** моделей для актуальных данных
- **Аналитика ML-моделей** - отчеты по эффективности, топ-фичам и точности

## Установка

1. Клонируйте репозиторий:
```
git clone https://github.com/your-username/crypto-signals-bot.git
cd crypto-signals-bot
```

2. Установите зависимости:
```
pip install -r requirements.txt
```

3. Настройте конфигурацию в `config.json` или создайте `.env` файл:
```
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key
```

## Использование

### Запуск бота
```
python crypto_signals_bot.py
```

### Обучение ML-моделей
```
python crypto_signals_bot.py --train-ml
```

### Бэктестинг
```
# Бэктестинг конкретной пары
python crypto_signals_bot.py --backtest BTC/USDT:USDT --start-date 2023-01-01

# Бэктестинг всех пар с высоким объемом
python crypto_signals_bot.py --backtest all --start-date 2023-01-01
```

### Генерация ML-отчета
```
python crypto_signals_bot.py --generate-ml-report
```

## Структура сигналов

Сигналы включают в себя:
- Символ и тип (long/short)
- Цену входа, TP, SL
- Леверидж
- Индикаторы, подтверждающие сигнал
- ML-прогноз и уверенность модели
- Контекст новостей

## Файлы и директории

- `crypto_signals_bot.py` - основной файл бота
- `ml_models.py` - модуль ML-моделей
- `config.json` - настройки бота
- `models/` - сохраненные ML-модели
- `cache/` - кэшированные данные OHLCV
- `signals_history.csv` - история сигналов
- `ml_analytics_report.xlsx` - отчет по ML-моделям

## Настройка параметров

Основные параметры в `config.json`:
- `signal_threshold` - порог срабатывания индикаторов (например, 0.52 = 12/23)
- `leverage` - стандартный леверидж
- `meme_leverage` - леверидж для мем-монет
- `ml_confidence_threshold` - минимальная уверенность ML-модели
- `min_ml_accuracy` - минимальная точность ML-модели для использования

## Лицензия

MIT 