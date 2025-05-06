import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import logging
from datetime import datetime
import joblib
import ta
from ta.trend import IchimokuIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# Configure logging
logger = logging.getLogger(__name__)

class ModelType:
    """Enum-like class for model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"

class CryptoMLPredictor:
    def __init__(self, symbol, timeframe='1h', model_dir='models', model_type=ModelType.RANDOM_FOREST):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = model_dir
        self.model_type = model_type
        
        # File paths for model storage
        base_name = f'{symbol.replace("/", "_")}_{timeframe}_{model_type}'
        self.model_path = os.path.join(model_dir, f'{base_name}_model.pkl')
        self.scaler_path = os.path.join(model_dir, f'{base_name}_scaler.pkl')
        self.feature_importance_path = os.path.join(model_dir, f'{base_name}_feature_importance.pkl')
        self.features_path = os.path.join(model_dir, f'{base_name}_features.pkl')
        self.metrics_path = os.path.join(model_dir, f'{base_name}_metrics.json')
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Default model parameters
        self.model = None
        self.scaler = None
        self.features = None
        self.feature_importance = None
        self.prediction_horizon = 3  # Predict price movement 3 candles ahead
        self.metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'last_training': None,
            'training_samples': 0,
            'feature_importance': {}
        }
        
        logger.info(f"Инициализирована ML модель для {symbol} (тип: {self.model_type})")
    
    def prepare_data(self, df):
        """Prepare data by adding features and creating target variable"""
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Calculate technical indicators as features
        self._add_features(data)
        
        # Create target variable: 1 if price goes up in next prediction_horizon candles, 0 otherwise
        data['target'] = (data['close'].shift(-self.prediction_horizon) > data['close']).astype(int)
        
        # Drop NaN values
        data = data.dropna()
        
        # Select only the features and target
        features_df = data[self.features]
        target = data['target']
        
        return features_df, target
    
    def _add_features(self, df):
        """Add technical indicators and other features to the dataframe"""
        # Price based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_3'] = df['close'].pct_change(3)
        df['price_change_5'] = df['close'].pct_change(5)
        
        # Volatility
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        
        # Trend
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        df['ema_trend'] = (df['ema_9'] > df['ema_21']).astype(int)
        df['ema_cross'] = ((df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))).astype(int)
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_cross'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        # ADX
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Momentum
        rsi = RSIIndicator(close=df['close'])
        df['rsi'] = rsi.rsi()
        df['rsi_diff'] = df['rsi'] - df['rsi'].shift(1)
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volume
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_ma'] = df['obv'].rolling(window=20).mean()
        
        # Distance from moving averages
        df['distance_ema9'] = (df['close'] - df['ema_9']) / df['close']
        df['distance_ema21'] = (df['close'] - df['ema_21']) / df['close']
        df['distance_ema50'] = (df['close'] - df['ema_50']) / df['close']
        
        # Price patterns
        df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['lower_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))).astype(int)
        
        # Volatility related
        df['atr_percent'] = df['atr'] / df['close'] * 100
        df['bb_squeeze'] = ((df['bb_upper'] - df['bb_lower']) / df['close']).rolling(window=20).std()
        
        # Feature list (all features we want to use)
        self.features = [
            'price_change', 'price_change_1', 'price_change_3', 'price_change_5',
            'atr', 'bb_width', 'atr_percent', 'bb_squeeze',
            'ema_trend', 'ema_cross', 'distance_ema9', 'distance_ema21', 'distance_ema50',
            'macd', 'macd_signal', 'macd_diff', 'macd_cross',
            'adx', 'adx_pos', 'adx_neg',
            'rsi', 'rsi_diff', 'stoch_k', 'stoch_d',
            'volume_change', 'volume_ratio', 'obv', 'obv_ma',
            'higher_high', 'lower_low'
        ]
        
        # Save features list
        joblib.dump(self.features, self.features_path)
        
        return df
    
    def train(self, df, test_size=0.2, random_state=42):
        """Train the model using the provided OHLCV dataframe"""
        try:
            logger.info(f"Начало обучения модели для {self.symbol}")
            
            # Prepare data
            X, y = self.prepare_data(df)
            if X is None or y is None:
                logger.error(f"Не удалось подготовить данные для {self.symbol}")
                return None
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Initialize and train model based on type
            if self.model_type == ModelType.RANDOM_FOREST:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1
                )
            elif self.model_type == ModelType.GRADIENT_BOOSTING:
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=random_state
                )
            elif self.model_type == ModelType.ENSEMBLE:
                rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
                gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
                self.model = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb)],
                    voting='soft'
                )
            
            # Train model
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.metrics['precision'] = report['weighted avg']['precision']
            self.metrics['recall'] = report['weighted avg']['recall']
            self.metrics['f1_score'] = report['weighted avg']['f1-score']
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.features, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                self.feature_importance = dict(zip(self.features, abs(self.model.coef_[0])))
            
            # Update metrics
            self.metrics['last_training'] = datetime.now().isoformat()
            self.metrics['training_samples'] = len(X_train)
            
            # Save everything
            self.save_model()
            
            logger.info(f"Модель {self.symbol} обучена. Точность: {self.metrics['accuracy']:.2f}")
            
            return {
                'test_accuracy': self.metrics['accuracy'],
                'metrics': self.metrics,
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели {self.symbol}: {e}")
            return None
    
    def predict(self, df):
        """Make prediction for the latest data point"""
        try:
            if self.model is None:
                if not self.load_model():
                    return None
            
            # Prepare data
            X, _ = self.prepare_data(df)
            
            if X is None:
                return None
            
            # Make prediction
            prob = self.model.predict_proba(X)[-1][1]
            
            prediction = {
                'direction': 'up' if prob > 0.5 else 'down',
                'confidence': prob if prob > 0.5 else 1 - prob,
                'probability': prob,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Предсказание для {self.symbol}: {prediction['direction']} (уверенность: {prediction['confidence']:.2f})")
            return prediction
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании для {self.symbol}: {e}")
            return None
    
    def save_model(self):
        """Save model and related data"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.features, self.features_path)
            joblib.dump(self.feature_importance, self.feature_importance_path)
            
            # Save metrics
            import json
            with open(self.metrics_path, 'w') as f:
                json.dump(self.metrics, f)
                
            logger.info(f"Модель {self.symbol} сохранена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели {self.symbol}: {e}")
            return False
    
    def load_model(self):
        """Load model and related data"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Модель для {self.symbol} не найдена")
                return False
            
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.features = joblib.load(self.features_path)
            self.feature_importance = joblib.load(self.feature_importance_path)
            
            # Load metrics
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                    
            logger.info(f"Модель {self.symbol} загружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {self.symbol}: {e}")
            return False

    def test_model(self, df):
        """Тестирование работоспособности модели"""
        try:
            # Подготовка тестовых данных
            X, y = self.prepare_data(df)
            if X is None or y is None:
                logger.error(f"Не удалось подготовить данные для тестирования модели {self.symbol}")
                return False
                
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Создание и обучение модели
            if self.model_type == ModelType.RANDOM_FOREST:
                test_model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.model_type == ModelType.GRADIENT_BOOSTING:
                test_model = GradientBoostingClassifier(random_state=42)
            elif self.model_type == ModelType.ENSEMBLE:
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                gb = GradientBoostingClassifier(random_state=42)
                test_model = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb)],
                    voting='soft'
                )
            
            # Обучение и оценка
            test_model.fit(X_train, y_train)
            score = test_model.score(X_test, y_test)
            
            logger.info(f"Тест модели {self.symbol} успешен. Точность: {score:.2f}")
            return score >= 0.6  # Минимальная приемлемая точность
            
        except Exception as e:
            logger.error(f"Ошибка при тестировании модели {self.symbol}: {e}")
            return False

    def check_model_health(self):
        """Проверка здоровья модели"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Модель для {self.symbol} не найдена")
                return False
                
            # Проверяем возраст модели
            model_age = datetime.now().timestamp() - os.path.getmtime(self.model_path)
            if model_age > 7 * 24 * 3600:  # Старше 7 дней
                logger.warning(f"Модель для {self.symbol} устарела (возраст: {model_age/3600:.1f} часов)")
                return False
                
            # Загружаем модель
            if not self.load_model():
                logger.error(f"Не удалось загрузить модель для {self.symbol}")
                return False
                
            # Проверяем наличие всех необходимых файлов
            required_files = [self.model_path, self.scaler_path, self.feature_importance_path]
            for file in required_files:
                if not os.path.exists(file):
                    logger.error(f"Отсутствует необходимый файл: {file}")
                    return False
            
            logger.info(f"Проверка здоровья модели {self.symbol} успешна")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при проверке здоровья модели {self.symbol}: {e}")
            return False

# Function to convert OHLCV data to pandas DataFrame
def ohlcv_to_dataframe(ohlcv_data):
    """Convert OHLCV data from ccxt to pandas DataFrame"""
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Batch training function for multiple symbols
async def batch_train_models(symbol_data_dict, timeframe='1h', model_dir='models', model_type=ModelType.RANDOM_FOREST):
    """Train ML models for multiple symbols in batch
    
    Args:
        symbol_data_dict: Dictionary with symbol as key and OHLCV dataframe as value
        timeframe: Timeframe for the model
        model_dir: Directory to save models
        model_type: Type of model to train (random_forest, gradient_boosting, ensemble, lstm)
    
    Returns:
        Dictionary with training results for each symbol
    """
    results = {}
    
    for symbol, ohlcv_data in symbol_data_dict.items():
        try:
            # Convert to dataframe if not already
            if not isinstance(ohlcv_data, pd.DataFrame):
                df = ohlcv_to_dataframe(ohlcv_data)
            else:
                df = ohlcv_data
                
            # Check if we have enough data
            min_samples = 100 if model_type != ModelType.LSTM else 150
            if len(df) < min_samples:
                logger.warning(f"Not enough data for {symbol} to train ML model (min {min_samples} candles)")
                continue
                
            # Train model
            model = CryptoMLPredictor(symbol, timeframe, model_dir, model_type=model_type)
            training_results = model.train(df)
            
            if training_results is None:
                continue
                
            results[symbol] = {
                'success': True,
                'model_type': model_type,
                'accuracy': training_results['test_accuracy'],
                'top_features': sorted(
                    training_results.get('feature_importance', {}).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5] if training_results.get('feature_importance') else []
            }
            
            logger.info(f"Successfully trained {model_type} model for {symbol} with accuracy {training_results['test_accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            results[symbol] = {'success': False, 'error': str(e), 'model_type': model_type}
    
    return results 