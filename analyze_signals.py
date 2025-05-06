import os
import pandas as pd
import logging
from datetime import datetime, timedelta
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signal_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_signal_files(log_dir='logs/'):
    """Load all signal files from logs directory"""
    try:
        logger.info(f"Loading signal files from {log_dir}")
        
        if not os.path.exists(log_dir):
            logger.warning(f"Logs directory {log_dir} does not exist")
            return pd.DataFrame()
            
        # Find all signal files
        signal_files = glob.glob(os.path.join(log_dir, 'signals_*.csv'))
        if not signal_files:
            logger.warning("No signal files found")
            return pd.DataFrame()
            
        logger.info(f"Found {len(signal_files)} signal files")
        
        # Load and combine all files
        dfs = []
        for file in signal_files:
            try:
                df = pd.read_csv(file)
                df['source_file'] = os.path.basename(file)
                dfs.append(df)
                logger.debug(f"Loaded {len(df)} signals from {file}")
            except Exception as e:
                logger.error(f"Error loading file {file}: {e}")
                continue
                
        if not dfs:
            logger.warning("No valid signal files loaded")
            return pd.DataFrame()
            
        # Combine all dataframes
        signals_df = pd.concat(dfs, ignore_index=True)
        
        # Convert timestamp to datetime
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        
        # Sort by timestamp
        signals_df = signals_df.sort_values('timestamp')
        
        logger.info(f"Loaded total of {len(signals_df)} signals")
        return signals_df
        
    except Exception as e:
        logger.error(f"Error loading signal files: {e}")
        return pd.DataFrame()

def analyze_signal_results(signals_df, ohlcv_data):
    """Analyze signal results using OHLCV data"""
    try:
        if signals_df.empty:
            logger.warning("No signals to analyze")
            return pd.DataFrame()
            
        logger.info("Analyzing signal results...")
        
        results = []
        for _, signal in signals_df.iterrows():
            try:
                # Get OHLCV data for the symbol
                symbol_data = ohlcv_data.get(signal['symbol'])
                if symbol_data is None:
                    logger.warning(f"No OHLCV data for {signal['symbol']}")
                    continue
                    
                # Convert to DataFrame if not already
                if not isinstance(symbol_data, pd.DataFrame):
                    symbol_data = pd.DataFrame(symbol_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'], unit='ms')
                    symbol_data = symbol_data.set_index('timestamp')
                
                # Get data after signal timestamp
                future_data = symbol_data[symbol_data.index > signal['timestamp']]
                if future_data.empty:
                    logger.debug(f"No future data for signal {signal['timestamp']} {signal['symbol']}")
                    continue
                
                # Initialize result
                result = signal.to_dict()
                result['win'] = None
                result['time_to_target'] = None
                result['profit_loss'] = None
                
                # Check if TP or SL was hit
                if signal['type'].lower() == 'long':
                    # For long positions
                    tp_hit = future_data['high'] >= signal['tp']
                    sl_hit = future_data['low'] <= signal['sl']
                    
                    # Calculate potential profit/loss
                    if signal['entry'] > 0:  # Avoid division by zero
                        result['profit_potential'] = (signal['tp'] - signal['entry']) / signal['entry'] * 100
                        result['loss_risk'] = (signal['entry'] - signal['sl']) / signal['entry'] * 100
                else:
                    # For short positions
                    tp_hit = future_data['low'] <= signal['tp']
                    sl_hit = future_data['high'] >= signal['sl']
                    
                    # Calculate potential profit/loss
                    if signal['entry'] > 0:  # Avoid division by zero
                        result['profit_potential'] = (signal['entry'] - signal['tp']) / signal['entry'] * 100
                        result['loss_risk'] = (signal['sl'] - signal['entry']) / signal['entry'] * 100
                
                # Find which happened first
                if tp_hit.any() or sl_hit.any():
                    tp_idx = tp_hit.idxmax() if tp_hit.any() else None
                    sl_idx = sl_hit.idxmax() if sl_hit.any() else None
                    
                    if tp_idx and (not sl_idx or tp_idx < sl_idx):
                        # TP hit first
                        result['win'] = 1
                        result['time_to_target'] = (tp_idx - signal['timestamp']).total_seconds() / 3600
                        result['profit_loss'] = result['profit_potential']
                    elif sl_idx:
                        # SL hit first
                        result['win'] = 0
                        result['time_to_target'] = (sl_idx - signal['timestamp']).total_seconds() / 3600
                        result['profit_loss'] = -result['loss_risk']
                
                # Add market conditions
                result['market_volume'] = symbol_data.loc[signal['timestamp']:]['volume'].mean()
                result['price_volatility'] = symbol_data.loc[signal['timestamp']:]['close'].pct_change().std() * 100
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing signal {signal['timestamp']} {signal['symbol']}: {e}")
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate statistics
        if not results_df.empty:
            total_signals = len(results_df)
            completed_signals = results_df['win'].notna().sum()
            win_rate = (results_df['win'] == 1).sum() / completed_signals if completed_signals > 0 else 0
            
            logger.info(f"\nAnalysis completed:")
            logger.info(f"Total signals: {total_signals}")
            logger.info(f"Completed signals: {completed_signals}")
            logger.info(f"Win rate: {win_rate:.2%}")
            
            # Calculate more statistics
            avg_profit = results_df[results_df['win'] == 1]['profit_loss'].mean()
            avg_loss = abs(results_df[results_df['win'] == 0]['profit_loss'].mean())
            profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            logger.info(f"\nProfitability metrics:")
            logger.info(f"Average profit: {avg_profit:.2f}%")
            logger.info(f"Average loss: {avg_loss:.2f}%")
            logger.info(f"Profit factor: {profit_factor:.2f}")
            
            # Results by symbol
            by_symbol = results_df.groupby('symbol').agg({
                'win': ['count', 'mean'],
                'time_to_target': 'mean',
                'profit_loss': ['mean', 'std'],
                'market_volume': 'mean',
                'price_volatility': 'mean'
            }).round(3)
            
            logger.info("\nResults by symbol:")
            logger.info(by_symbol)
            
            # Results by signal type
            by_type = results_df.groupby('type').agg({
                'win': ['count', 'mean'],
                'profit_loss': ['mean', 'std']
            }).round(3)
            
            logger.info("\nResults by signal type:")
            logger.info(by_type)
            
            # Create visualizations
            create_analysis_plots(results_df)
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error analyzing signal results: {e}")
        return pd.DataFrame()

def create_analysis_plots(results_df):
    """Create visualization plots for signal analysis"""
    try:
        # Create a directory for plots
        os.makedirs('analysis_plots', exist_ok=True)
        
        # 1. Win rate by symbol
        plt.figure(figsize=(12, 6))
        win_rates = results_df.groupby('symbol')['win'].mean().sort_values(ascending=False)
        win_rates.plot(kind='bar')
        plt.title('Win Rate by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Win Rate')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_plots/win_rate_by_symbol.png')
        plt.close()
        
        # 2. Profit/Loss distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(data=results_df, x='profit_loss', hue='type', bins=30)
        plt.title('Profit/Loss Distribution')
        plt.xlabel('Profit/Loss %')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_plots/profit_loss_distribution.png')
        plt.close()
        
        # 3. Time to target distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x='symbol', y='time_to_target', hue='win')
        plt.title('Time to Target by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Hours')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_plots/time_to_target.png')
        plt.close()
        
        # 4. Win rate over time
        plt.figure(figsize=(12, 6))
        results_df['date'] = results_df['timestamp'].dt.date
        daily_win_rate = results_df.groupby('date')['win'].mean()
        daily_win_rate.plot(kind='line', marker='o')
        plt.title('Win Rate Over Time')
        plt.xlabel('Date')
        plt.ylabel('Win Rate')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_plots/win_rate_over_time.png')
        plt.close()
        
        # 5. Correlation matrix
        plt.figure(figsize=(10, 8))
        numeric_cols = ['profit_loss', 'time_to_target', 'market_volume', 'price_volatility', 'profit_potential', 'loss_risk']
        correlation_matrix = results_df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('analysis_plots/correlation_matrix.png')
        plt.close()
        
        # 6. Profit/Loss by symbol and type
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x='symbol', y='profit_loss', hue='type')
        plt.title('Profit/Loss by Symbol and Type')
        plt.xlabel('Symbol')
        plt.ylabel('Profit/Loss %')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('analysis_plots/profit_loss_by_symbol.png')
        plt.close()
        
        # 7. Market conditions impact
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.scatterplot(data=results_df, x='market_volume', y='profit_loss', hue='win', ax=ax1)
        ax1.set_title('Profit/Loss vs Market Volume')
        ax1.set_xlabel('Market Volume')
        ax1.set_ylabel('Profit/Loss %')
        ax1.grid(True)
        
        sns.scatterplot(data=results_df, x='price_volatility', y='profit_loss', hue='win', ax=ax2)
        ax2.set_title('Profit/Loss vs Price Volatility')
        ax2.set_xlabel('Price Volatility %')
        ax2.set_ylabel('Profit/Loss %')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('analysis_plots/market_conditions_impact.png')
        plt.close()
        
        logger.info("Analysis plots saved in 'analysis_plots' directory")
        
    except Exception as e:
        logger.error(f"Error creating analysis plots: {e}")

def save_results(results_df, output_file='signals_history_with_win.xlsx'):
    """Save analysis results to Excel file"""
    try:
        if results_df.empty:
            logger.warning("No results to save")
            return False
            
        # Ensure all timestamps are in datetime format
        for col in results_df.columns:
            if 'time' in col.lower() and results_df[col].dtype == 'object':
                try:
                    results_df[col] = pd.to_datetime(results_df[col])
                except:
                    pass
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save main results
            results_df.to_excel(writer, sheet_name='Signals', index=False)
            
            # Create summary sheet
            summary_data = {
                'Metric': [
                    'Total Signals',
                    'Completed Signals',
                    'Win Rate',
                    'Average Profit',
                    'Average Loss',
                    'Profit Factor',
                    'Average Time to Target (hours)',
                    'Average Volume',
                    'Average Volatility'
                ],
                'Value': [
                    len(results_df),
                    results_df['win'].notna().sum(),
                    f"{(results_df['win'] == 1).mean():.2%}",
                    f"{results_df[results_df['win'] == 1]['profit_loss'].mean():.2f}%",
                    f"{abs(results_df[results_df['win'] == 0]['profit_loss'].mean()):.2f}%",
                    f"{abs(results_df[results_df['win'] == 1]['profit_loss'].mean() / results_df[results_df['win'] == 0]['profit_loss'].mean()):.2f}",
                    f"{results_df['time_to_target'].mean():.2f}",
                    f"{results_df['market_volume'].mean():.0f}",
                    f"{results_df['price_volatility'].mean():.2f}%"
                ]
            }
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Add symbol statistics
            symbol_stats = results_df.groupby('symbol').agg({
                'win': ['count', 'mean'],
                'profit_loss': ['mean', 'std'],
                'time_to_target': 'mean'
            }).round(3)
            symbol_stats.to_excel(writer, sheet_name='Symbol Stats')
            
            # Add signal type statistics
            type_stats = results_df.groupby('type').agg({
                'win': ['count', 'mean'],
                'profit_loss': ['mean', 'std']
            }).round(3)
            type_stats.to_excel(writer, sheet_name='Type Stats')
        
        logger.info(f"Results saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

def main():
    """Main function to analyze signal history"""
    try:
        # Load signal files
        signals_df = load_signal_files()
        if signals_df.empty:
            logger.error("No signals to analyze")
            return
        
        # Load OHLCV data for all symbols
        symbols = signals_df['symbol'].unique()
        ohlcv_data = {}
        
        # For now, we'll use test data
        from test_data import generate_test_ohlcv
        for symbol in symbols:
            ohlcv_data[symbol] = generate_test_ohlcv(symbol, num_candles=1000)
        
        # Analyze results
        results_df = analyze_signal_results(signals_df, ohlcv_data)
        if results_df.empty:
            logger.error("Analysis failed")
            return
        
        # Save results
        save_results(results_df)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 