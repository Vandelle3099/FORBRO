"""
Pattern Recognition Module for Crypto Signals Bot
Provides functions to detect chart patterns in price data
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
from scipy.signal import argrelextrema
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import talib for pattern recognition, fall back to custom implementations if not available
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("TA-Lib is available. Using TA-Lib for pattern recognition")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib is not available. Using custom pattern recognition")

# Candlestick Patterns
class CandlestickPatterns:
    """Detect candlestick patterns in price data"""
    
    @staticmethod
    def detect_patterns(df):
        """Detect all supported candlestick patterns in the dataframe
        
        Args:
            df: Pandas DataFrame with OHLC data
            
        Returns:
            DataFrame with pattern columns
        """
        result = df.copy()
        
        if TALIB_AVAILABLE:
            # Use TA-Lib for pattern detection
            return CandlestickPatterns._detect_talib_patterns(result)
        else:
            # Use custom implementations
            return CandlestickPatterns._detect_custom_patterns(result)
    
    @staticmethod
    def _detect_talib_patterns(df):
        """Detect patterns using TA-Lib"""
        # Bullish patterns
        df['cdl_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['cdl_inverted_hammer'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['cdl_engulfing_bullish'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['cdl_morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdl_bullish_harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
        df['cdl_piercing'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
        df['cdl_three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
        
        # Bearish patterns
        df['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdl_hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
        df['cdl_engulfing_bearish'] = -1 * (talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) < 0).astype(int)
        df['cdl_evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['cdl_bearish_harami'] = -1 * (talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close']) < 0).astype(int)
        df['cdl_three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
        
        # Normalize pattern signals to boolean
        pattern_columns = [col for col in df.columns if col.startswith('cdl_')]
        for col in pattern_columns:
            df[col] = (df[col] != 0).astype(int)
            
        return df
    
    @staticmethod
    def _detect_custom_patterns(df):
        """Detect patterns using custom implementations"""
        # Bullish patterns
        df['cdl_hammer'] = CandlestickPatterns._detect_hammer(df)
        df['cdl_engulfing_bullish'] = CandlestickPatterns._detect_bullish_engulfing(df)
        df['cdl_morning_star'] = CandlestickPatterns._detect_morning_star(df)
        
        # Bearish patterns
        df['cdl_shooting_star'] = CandlestickPatterns._detect_shooting_star(df)
        df['cdl_engulfing_bearish'] = CandlestickPatterns._detect_bearish_engulfing(df)
        df['cdl_evening_star'] = CandlestickPatterns._detect_evening_star(df)
        
        return df
    
    @staticmethod
    def _detect_hammer(df):
        """Detect hammer pattern - custom implementation"""
        # Hammer has a small body, long lower shadow, and small or no upper shadow
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Calculate ratios
        body_to_range = body_size / (df['high'] - df['low'])
        lower_shadow_to_body = lower_shadow / body_size.replace(0, 0.000001)
        upper_shadow_to_body = upper_shadow / body_size.replace(0, 0.000001)
        
        # Conditions for hammer
        hammer = (
            (body_to_range < 0.3) &  # Small body
            (lower_shadow_to_body > 2) &  # Long lower shadow
            (upper_shadow_to_body < 0.5) &  # Small upper shadow
            (df['close'] > df['open'])  # Bullish
        ).astype(int)
        
        return hammer
    
    @staticmethod
    def _detect_shooting_star(df):
        """Detect shooting star pattern - custom implementation"""
        # Shooting star has a small body, long upper shadow, and small or no lower shadow
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Calculate ratios
        body_to_range = body_size / (df['high'] - df['low'])
        upper_shadow_to_body = upper_shadow / body_size.replace(0, 0.000001)
        lower_shadow_to_body = lower_shadow / body_size.replace(0, 0.000001)
        
        # Conditions for shooting star
        shooting_star = (
            (body_to_range < 0.3) &  # Small body
            (upper_shadow_to_body > 2) &  # Long upper shadow
            (lower_shadow_to_body < 0.5) &  # Small lower shadow
            (df['close'] < df['open'])  # Bearish
        ).astype(int)
        
        return shooting_star
    
    @staticmethod
    def _detect_bullish_engulfing(df):
        """Detect bullish engulfing pattern - custom implementation"""
        bullish_engulfing = (
            (df['open'].shift(1) > df['close'].shift(1)) &  # Previous candle is bearish
            (df['close'] > df['open']) &  # Current candle is bullish
            (df['open'] <= df['close'].shift(1)) &  # Current open is lower than or equal to previous close
            (df['close'] >= df['open'].shift(1))  # Current close is higher than or equal to previous open
        ).astype(int)
        
        return bullish_engulfing
    
    @staticmethod
    def _detect_bearish_engulfing(df):
        """Detect bearish engulfing pattern - custom implementation"""
        bearish_engulfing = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
            (df['open'] > df['close']) &  # Current candle is bearish
            (df['open'] >= df['close'].shift(1)) &  # Current open is higher than or equal to previous close
            (df['close'] <= df['open'].shift(1))  # Current close is lower than or equal to previous open
        ).astype(int)
        
        return bearish_engulfing
    
    @staticmethod
    def _detect_morning_star(df):
        """Detect morning star pattern - custom implementation"""
        # Morning star is a three-candle pattern
        # First candle is a long bearish candle
        # Second candle is a small-bodied candle (star) that gaps down
        # Third candle is a bullish candle that closes above the midpoint of the first candle
        
        # Calculate body sizes
        body_size = abs(df['close'] - df['open'])
        body_size_pct = body_size / df['close'] * 100
        
        # First candle: bearish with large body
        first_bearish = (df['close'].shift(2) < df['open'].shift(2)) & (body_size_pct.shift(2) > 1.0)
        
        # Second candle: small body with gap down
        second_small = body_size_pct.shift(1) < 0.5
        gap_down = df[['open', 'close']].shift(1).max(axis=1) < df[['open', 'close']].shift(2).min(axis=1)
        
        # Third candle: bullish with close above midpoint of first candle
        third_bullish = df['close'] > df['open']
        third_close_high = df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2
        
        # Combine conditions
        morning_star = (
            first_bearish &
            second_small &
            gap_down &
            third_bullish &
            third_close_high
        ).astype(int)
        
        return morning_star
    
    @staticmethod
    def _detect_evening_star(df):
        """Detect evening star pattern - custom implementation"""
        # Evening star is a three-candle pattern
        # First candle is a long bullish candle
        # Second candle is a small-bodied candle (star) that gaps up
        # Third candle is a bearish candle that closes below the midpoint of the first candle
        
        # Calculate body sizes
        body_size = abs(df['close'] - df['open'])
        body_size_pct = body_size / df['close'] * 100
        
        # First candle: bullish with large body
        first_bullish = (df['close'].shift(2) > df['open'].shift(2)) & (body_size_pct.shift(2) > 1.0)
        
        # Second candle: small body with gap up
        second_small = body_size_pct.shift(1) < 0.5
        gap_up = df[['open', 'close']].shift(1).min(axis=1) > df[['open', 'close']].shift(2).max(axis=1)
        
        # Third candle: bearish with close below midpoint of first candle
        third_bearish = df['close'] < df['open']
        third_close_low = df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2
        
        # Combine conditions
        evening_star = (
            first_bullish &
            second_small &
            gap_up &
            third_bearish &
            third_close_low
        ).astype(int)
        
        return evening_star
    
    @staticmethod
    def get_pattern_signals(df):
        """Get bullish and bearish signals from detected patterns
        
        Args:
            df: DataFrame with pattern columns
            
        Returns:
            tuple: (bullish_score, bearish_score, active_patterns)
        """
        # Bullish patterns
        bullish_patterns = [
            'cdl_hammer', 'cdl_inverted_hammer', 'cdl_engulfing_bullish',
            'cdl_morning_star', 'cdl_bullish_harami', 'cdl_piercing',
            'cdl_three_white_soldiers'
        ]
        
        # Bearish patterns
        bearish_patterns = [
            'cdl_shooting_star', 'cdl_hanging_man', 'cdl_engulfing_bearish',
            'cdl_evening_star', 'cdl_bearish_harami', 'cdl_three_black_crows'
        ]
        
        # Filter for available columns
        bullish_patterns = [p for p in bullish_patterns if p in df.columns]
        bearish_patterns = [p for p in bearish_patterns if p in df.columns]
        
        # Get the last row for current signals
        last_row = df.iloc[-1]
        
        # Calculate the scores
        bullish_score = sum(last_row[p] for p in bullish_patterns)
        bearish_score = sum(last_row[p] for p in bearish_patterns)
        
        # Get active patterns
        active_bullish = [p.replace('cdl_', '') for p in bullish_patterns if last_row[p] > 0]
        active_bearish = [p.replace('cdl_', '') for p in bearish_patterns if last_row[p] > 0]
        
        active_patterns = {
            'bullish': active_bullish,
            'bearish': active_bearish
        }
        
        return bullish_score, bearish_score, active_patterns

# Chart Patterns
class PatternType(Enum):
    """Enum for pattern types"""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    CUP_AND_HANDLE = "cup_and_handle"
    ROUNDED_BOTTOM = "rounded_bottom"
    ROUNDED_TOP = "rounded_top"

class ChartPatterns:
    """Detect chart patterns in price data"""
    
    @staticmethod
    def find_extrema(df, column='close', order=5):
        """Find local maxima and minima in the data
        
        Args:
            df: DataFrame with price data
            column: Column to find extrema in
            order: Order of the extrema (how many points on each side to compare)
            
        Returns:
            tuple: (maxima_indices, minima_indices)
        """
        # Find local maxima and minima
        maxima_indices = argrelextrema(df[column].values, np.greater, order=order)[0]
        minima_indices = argrelextrema(df[column].values, np.less, order=order)[0]
        
        return maxima_indices, minima_indices
    
    @staticmethod
    def detect_patterns(df, lookback_period=100):
        """Detect chart patterns in the data
        
        Args:
            df: DataFrame with price data
            lookback_period: Number of candles to look back for pattern detection
            
        Returns:
            dict: Dictionary with detected patterns
        """
        # Consider only the last `lookback_period` candles
        if len(df) > lookback_period:
            df_subset = df.iloc[-lookback_period:]
        else:
            df_subset = df.copy()
        
        # Find extrema points
        maxima_indices, minima_indices = ChartPatterns.find_extrema(df_subset)
        
        # Initialize results
        patterns = {
            'head_and_shoulders': False,
            'inverse_head_and_shoulders': False,
            'double_top': False,
            'double_bottom': False,
            'ascending_triangle': False,
            'descending_triangle': False,
            'symmetrical_triangle': False,
            'rising_wedge': False,
            'falling_wedge': False
        }
        
        # Check for head and shoulders pattern
        if len(maxima_indices) >= 3:
            # Get the high points
            highs = df_subset.iloc[maxima_indices]['high'].values
            
            # Check for head and shoulders pattern (3 peaks with middle peak higher)
            # Iterate through potential formations (need at least 3 peaks)
            for i in range(len(maxima_indices) - 2):
                if (highs[i] < highs[i+1] > highs[i+2]) and abs(highs[i] - highs[i+2]) / highs[i] < 0.05:
                    patterns['head_and_shoulders'] = True
                    break
        
        # Check for inverse head and shoulders pattern
        if len(minima_indices) >= 3:
            # Get the low points
            lows = df_subset.iloc[minima_indices]['low'].values
            
            # Check for inverse head and shoulders pattern (3 valleys with middle valley lower)
            for i in range(len(minima_indices) - 2):
                if (lows[i] > lows[i+1] < lows[i+2]) and abs(lows[i] - lows[i+2]) / lows[i] < 0.05:
                    patterns['inverse_head_and_shoulders'] = True
                    break
        
        # Check for double top pattern
        if len(maxima_indices) >= 2:
            # Get the high points
            highs = df_subset.iloc[maxima_indices]['high'].values
            
            # Check for double top (2 peaks at similar levels)
            for i in range(len(maxima_indices) - 1):
                if abs(highs[i] - highs[i+1]) / highs[i] < 0.02:
                    patterns['double_top'] = True
                    break
        
        # Check for double bottom pattern
        if len(minima_indices) >= 2:
            # Get the low points
            lows = df_subset.iloc[minima_indices]['low'].values
            
            # Check for double bottom (2 valleys at similar levels)
            for i in range(len(minima_indices) - 1):
                if abs(lows[i] - lows[i+1]) / lows[i] < 0.02:
                    patterns['double_bottom'] = True
                    break
        
        # Check for ascending triangle
        if len(maxima_indices) >= 2 and len(minima_indices) >= 2:
            # Get the high and low points
            highs = df_subset.iloc[maxima_indices]['high'].values
            lows = df_subset.iloc[minima_indices]['low'].values
            
            # Ascending triangle has flat top (resistance) and rising bottom (support)
            flat_top = abs(np.max(highs) - np.min(highs)) / np.max(highs) < 0.02
            rising_bottom = np.all(np.diff(lows) > 0)
            
            if flat_top and rising_bottom:
                patterns['ascending_triangle'] = True
        
        # Check for descending triangle
        if len(maxima_indices) >= 2 and len(minima_indices) >= 2:
            # Get the high and low points
            highs = df_subset.iloc[maxima_indices]['high'].values
            lows = df_subset.iloc[minima_indices]['low'].values
            
            # Descending triangle has falling top (resistance) and flat bottom (support)
            falling_top = np.all(np.diff(highs) < 0)
            flat_bottom = abs(np.max(lows) - np.min(lows)) / np.max(lows) < 0.02
            
            if falling_top and flat_bottom:
                patterns['descending_triangle'] = True
        
        # Check for symmetrical triangle
        if len(maxima_indices) >= 2 and len(minima_indices) >= 2:
            # Get the high and low points
            highs = df_subset.iloc[maxima_indices]['high'].values
            lows = df_subset.iloc[minima_indices]['low'].values
            
            # Symmetrical triangle has falling top (resistance) and rising bottom (support)
            falling_top = np.all(np.diff(highs) < 0)
            rising_bottom = np.all(np.diff(lows) > 0)
            
            if falling_top and rising_bottom:
                patterns['symmetrical_triangle'] = True
                
        # Return detected patterns
        return patterns
    
    @staticmethod
    def get_pattern_signals(df, lookback_period=100):
        """Get bullish and bearish signals from detected chart patterns
        
        Args:
            df: DataFrame with price data
            lookback_period: Number of candles to look back for pattern detection
            
        Returns:
            tuple: (bullish_score, bearish_score, active_patterns)
        """
        # Detect patterns
        patterns = ChartPatterns.detect_patterns(df, lookback_period)
        
        # Bullish patterns
        bullish_patterns = [
            'inverse_head_and_shoulders',
            'double_bottom',
            'ascending_triangle',
            'falling_wedge'
        ]
        
        # Bearish patterns
        bearish_patterns = [
            'head_and_shoulders',
            'double_top',
            'descending_triangle',
            'rising_wedge'
        ]
        
        # Calculate scores
        bullish_score = sum(1 for p in bullish_patterns if patterns.get(p, False))
        bearish_score = sum(1 for p in bearish_patterns if patterns.get(p, False))
        
        # Get active patterns
        active_bullish = [p for p in bullish_patterns if patterns.get(p, False)]
        active_bearish = [p for p in bearish_patterns if patterns.get(p, False)]
        
        active_patterns = {
            'bullish': active_bullish,
            'bearish': active_bearish
        }
        
        return bullish_score, bearish_score, active_patterns

# Integrate all pattern types for easy access
def analyze_patterns(df, lookback_period=100):
    """Analyze both candlestick and chart patterns
    
    Args:
        df: DataFrame with OHLC data
        lookback_period: Number of candles to look back for chart pattern detection
        
    Returns:
        dict: Dictionary with pattern analysis results
    """
    # Analyze candlestick patterns
    df_with_patterns = CandlestickPatterns.detect_patterns(df)
    cdl_bullish, cdl_bearish, cdl_patterns = CandlestickPatterns.get_pattern_signals(df_with_patterns)
    
    # Analyze chart patterns
    chart_bullish, chart_bearish, chart_patterns = ChartPatterns.get_pattern_signals(df, lookback_period)
    
    # Combine results
    return {
        'bullish_score': cdl_bullish + chart_bullish,
        'bearish_score': cdl_bearish + chart_bearish,
        'candlestick_patterns': cdl_patterns,
        'chart_patterns': chart_patterns,
        'total_patterns': {
            'bullish': cdl_patterns['bullish'] + chart_patterns['bullish'],
            'bearish': cdl_patterns['bearish'] + chart_patterns['bearish']
        }
    } 