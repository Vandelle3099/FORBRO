import os
import pandas as pd
import logging
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def read_signal_files(log_dir='logs/'):
    """Read all signal files from the logs directory"""
    logger.info(f"Reading signal files from {log_dir}")
    
    if not os.path.exists(log_dir):
        logger.info(f"Creating logs directory at {log_dir}")
        os.makedirs(log_dir)
        return pd.DataFrame()
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('signals_') and f.endswith('.csv')]
    
    if not log_files:
        logger.warning("No signal files found in logs directory")
        return pd.DataFrame()
    
    logger.info(f"Found {len(log_files)} signal files")
    
    dfs = []
    for file in log_files:
        try:
            file_path = os.path.join(log_dir, file)
            df = pd.read_csv(file_path)
            df['source_file'] = file  # Add source file information
            dfs.append(df)
            logger.info(f"Successfully read {file} with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
    
    if not dfs:
        logger.warning("No data could be read from signal files")
        return pd.DataFrame()
    
    signals_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total signals loaded: {len(signals_df)}")
    return signals_df

def prepare_data(df):
    """Prepare and clean the signal data"""
    if df.empty:
        return df
    
    logger.info("Preparing data...")
    
    # Standardize column names
    column_mapping = {
        'datetime': 'timestamp',
        'date': 'timestamp',
        'time': 'timestamp',
        'pair': 'symbol',
        'symbol_pair': 'symbol'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Ensure required columns exist
    required_columns = ['timestamp', 'symbol', 'type', 'entry', 'tp', 'sl']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return pd.DataFrame()
    
    try:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert numeric columns
        numeric_columns = ['entry', 'tp', 'sl']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Convert type to lowercase
        df['type'] = df['type'].str.lower()
        
        logger.info("Data preparation completed successfully")
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        return pd.DataFrame()
    
    return df

def filter_anomalies(df):
    """Filter out anomalous signals"""
    if df.empty:
        return df
    
    logger.info("Filtering anomalies...")
    original_count = len(df)
    
    # Remove rows with missing values
    df = df.dropna(subset=['timestamp', 'symbol', 'type', 'entry', 'tp', 'sl'])
    
    # Filter invalid signal types
    df = df[df['type'].isin(['long', 'short'])]
    
    # Filter invalid prices
    df = df[(df['entry'] > 0) & (df['tp'] > 0) & (df['sl'] > 0)]
    
    # Filter invalid TP/SL levels
    long_mask = (df['type'] == 'long') & (df['tp'] > df['entry']) & (df['sl'] < df['entry'])
    short_mask = (df['type'] == 'short') & (df['tp'] < df['entry']) & (df['sl'] > df['entry'])
    df = df[long_mask | short_mask]
    
    filtered_count = len(df)
    logger.info(f"Removed {original_count - filtered_count} anomalous signals")
    logger.info(f"Remaining signals after filtering: {filtered_count}")
    
    return df

def calculate_win(signal, ohlc_df):
    """Calculate if signal hit TP or SL first and time to target"""
    if ohlc_df.empty:
        return None, None
    
    entry_time = signal['timestamp']
    entry_price = signal['entry']
    tp = signal['tp']
    sl = signal['sl']
    signal_type = signal['type']
    
    # Filter OHLC data starting from entry time
    relevant_ohlc = ohlc_df[ohlc_df['timestamp'] >= entry_time].copy()
    
    if relevant_ohlc.empty:
        return None, None
    
    # Check if TP or SL was hit
    for idx, row in relevant_ohlc.iterrows():
        if signal_type == 'long':
            if row['high'] >= tp:
                time_to_target = (row['timestamp'] - entry_time).total_seconds() / 3600
                return 1, time_to_target
            if row['low'] <= sl:
                time_to_target = (row['timestamp'] - entry_time).total_seconds() / 3600
                return 0, time_to_target
        else:  # short
            if row['low'] <= tp:
                time_to_target = (row['timestamp'] - entry_time).total_seconds() / 3600
                return 1, time_to_target
            if row['high'] >= sl:
                time_to_target = (row['timestamp'] - entry_time).total_seconds() / 3600
                return 0, time_to_target
    
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

def analyze_results(signals_df):
    """Analyze results and print detailed statistics"""
    logger.info("\n=== Signal Analysis Results ===")
    
    total_signals = len(signals_df)
    logger.info(f"Total signals analyzed: {total_signals}")
    
    # Analyze by outcome
    tp_signals = signals_df[signals_df['win'] == 1]
    sl_signals = signals_df[signals_df['win'] == 0]
    pending_signals = signals_df[signals_df['win'].isnull()]
    
    tp_count = len(tp_signals)
    sl_count = len(sl_signals)
    pending_count = len(pending_signals)
    
    logger.info("\n--- Results by Category ---")
    logger.info(f"TP signals: {tp_count} ({tp_count/total_signals*100:.1f}%)")
    if not tp_signals.empty:
        logger.info("Examples of TP signals:")
        for _, row in tp_signals.head().iterrows():
            logger.info(f"  {row['symbol']} {row['type']} entry={row['entry']:.4f} TP hit after {row['time_to_target']:.1f}h")
    
    logger.info(f"\nSL signals: {sl_count} ({sl_count/total_signals*100:.1f}%)")
    if not sl_signals.empty:
        logger.info("Examples of SL signals:")
        for _, row in sl_signals.head().iterrows():
            logger.info(f"  {row['symbol']} {row['type']} entry={row['entry']:.4f} SL hit after {row['time_to_target']:.1f}h")
    
    logger.info(f"\nPending signals: {pending_count} ({pending_count/total_signals*100:.1f}%)")
    if not pending_signals.empty:
        logger.info("Examples of Pending signals:")
        for _, row in pending_signals.head().iterrows():
            logger.info(f"  {row['symbol']} {row['type']} entry={row['entry']:.4f}")
    
    # Calculate win rate for closed positions
    closed_positions = signals_df[signals_df['win'].notnull()]
    if not closed_positions.empty:
        win_rate = (len(tp_signals) / len(closed_positions)) * 100
        logger.info(f"\nOverall win rate: {win_rate:.2f}%")
        
        # Average time to target
        avg_time_to_tp = tp_signals['time_to_target'].mean()
        avg_time_to_sl = sl_signals['time_to_target'].mean()
        logger.info(f"Average time to TP: {avg_time_to_tp:.1f}h")
        logger.info(f"Average time to SL: {avg_time_to_sl:.1f}h")
    
    # Analysis by signal type
    logger.info("\n--- Analysis by Signal Type ---")
    for signal_type in ['long', 'short']:
        type_df = closed_positions[closed_positions['type'] == signal_type]
        if not type_df.empty:
            type_win_rate = (type_df['win'].mean() * 100)
            type_avg_time = type_df['time_to_target'].mean()
            logger.info(f"{signal_type.upper()}: {len(type_df)} signals, {type_win_rate:.1f}% win rate, avg time {type_avg_time:.1f}h")
    
    # Analysis by symbol
    logger.info("\n--- Analysis by Symbol ---")
    for symbol in signals_df['symbol'].unique():
        symbol_df = closed_positions[closed_positions['symbol'] == symbol]
        if not symbol_df.empty:
            symbol_win_rate = (symbol_df['win'].mean() * 100)
            symbol_avg_time = symbol_df['time_to_target'].mean()
            logger.info(f"{symbol}: {len(symbol_df)} signals, {symbol_win_rate:.1f}% win rate, avg time {symbol_avg_time:.1f}h")

def main():
    """Main function to analyze signal history"""
    try:
        # Read all signal files
        signals_df = read_signal_files()
        if signals_df.empty:
            logger.error("No signals to analyze")
            return
        
        # Prepare and clean data
        signals_df = prepare_data(signals_df)
        if signals_df.empty:
            logger.error("No valid signals after data preparation")
            return
        
        # Filter anomalies
        signals_df = filter_anomalies(signals_df)
        if signals_df.empty:
            logger.error("No valid signals after filtering anomalies")
            return
        
        # Get date range for OHLC data
        min_datetime = signals_df['timestamp'].min()
        max_datetime = signals_df['timestamp'].max() + pd.Timedelta(hours=24)
        
        # Get unique symbols
        symbols = signals_df['symbol'].unique()
        logger.info(f"Processing {len(symbols)} unique symbols")
        
        # Fetch OHLC data for all symbols
        from analyze import fetch_all_ohlc_data
        ohlc_cache = fetch_all_ohlc_data(symbols, min_datetime, max_datetime)
        
        # Process signals in parallel
        logger.info("Processing signals to determine outcomes...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            batch_size = 100
            futures = []
            
            for i in range(0, len(signals_df), batch_size):
                batch = signals_df.iloc[i:i+batch_size]
                future = executor.submit(process_signals_batch, batch, ohlc_cache)
                futures.append(future)
            
            # Collect results
            all_results = []
            all_times = []
            for future in as_completed(futures):
                results, times = future.result()
                all_results.extend(results)
                all_times.extend(times)
        
        # Add results to DataFrame
        signals_df['win'] = all_results
        signals_df['time_to_target'] = all_times
        
        # Analyze results
        analyze_results(signals_df)
        
        # Save results to Excel
        output_file = 'signals_history_with_win.xlsx'
        signals_df.to_excel(output_file, index=False)
        logger.info(f"\nResults saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)

if __name__ == "__main__":
    main() 