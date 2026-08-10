import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import argparse
import os

def process_data(input_file, output_file, time_step_sec, price_step, percentile_grid_size):
    # 1. Load data
    print(f"Loading file: {input_file}")
    df = pd.read_csv(input_file, sep='\t')

    # Convert TRADETIME and TRADETIME_MSEC to a single datetime object
    # We assume the date is from the filename if possible, but here we'll just use a dummy date 
    # since TRADETIME doesn't have it. However, for grid alignment (hours), we need a real date.
    # Let's try to extract date from file name if it matches 'YYYY.MM.DD.txt'
    try:
        date_str = os.path.basename(input_file).split('.')[0]
        base_date = datetime.strptime(date_str, '%Y.%m.%d')
    except:
        base_date = datetime(2026, 8, 10)

    def parse_time(row):
        t = datetime.strptime(row['TRADETIME'], '%H:%M:%S')
        return base_date + timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, milliseconds=row['TRADETIME_MSEC'])

    df['timestamp'] = df.apply(parse_time, axis=1)
    df['price'] = pd.to_numeric(df['PRICE'], errors='coerce')
    df['BUYSELL'] = df['BUYSELL'].str.upper()

    # Define grid parameters
    def get_grid_coords(row):
        # Time grid: floor to nearest time_step_sec
        t_seconds = row['timestamp'].timestamp()
        grid_t = (t_seconds // time_step_sec) * time_step_sec
        # Price grid: floor/ceil? Let's use floor for consistency
        grid_p = (row['price'] // price_step) * price_step
        return pd.to_datetime(grid_t, unit='s'), grid_p

    df[['grid_t', 'grid_p']] = df.apply(lambda r: pd.Series(get_grid_coords(r)), axis=1)

    def analyze_side(side_name, palette):
        print(f"Processing {side_name}...")
        side_df = df[df['BUYSELL'] == side_name].copy()
        if side_df.empty:
            return None

        # Group by grid cell
        grouped = side_df.groupby(['grid_t', 'grid_p'])
        
        cells = []
        values = []

        for (gt, gp), group in grouped:
            # 3. Differentiate prices and calc mean
            prices = group['price'].sort_values().values
            if len(prices) > 1:
                diffs = np.diff(prices)
                val = np.mean(diffs)
            else:
                val = 0.0 # Or skip if we need at least 2 prices to differentiate
            
            cells.append({'grid_t': gt, 'grid_p': gp, 'value': val})
            values.append(val)

        if not cells:
            return None

        cells_df = pd.DataFrame(cells)
        vals = np.array(values)

        # 5. Quantiles
        # We want PERCENTILE_GRID_SIZE number of bins (e.g. if size is 10, we have deciles)
        # Percentile boundaries
        quantiles = np.linspace(0, 100, percentile_grid_size + 1)
        bins = np.percentile(vals, quantiles)
        # Handle duplicate bin edges
        bins = np.unique(bins)
        
        # 6. Assign quantile index
        cells_df['quantile_idx'] = np.digitize(cells_df['value'], bins) - 1
        # Clip to ensure max value falls into last bin
        cells_df['quantile_idx'] = cells_df['quantile_idx'].clip(0, len(bins) - 2)

        # 7. Assign color
        cmap = plt.get_cmap(palette)
        # Normalize quantile index to [0, 1] for cmap
        num_colors = len(bins) - 1
        cells_df['color'] = cells_df['quantile_idx'].apply(lambda x: cmap(x / (num_colors if num_colors > 0 else 1)))

        return cells_df

    buy_cells = analyze_side('BUY', 'viridis')
    sell_cells = analyze_side('SELL', 'inferno')

    # Plotting
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    def plot_cells(ax, cells, title):
        if cells is None or cells.empty:
            ax.text(0.5, 0.5, f"No {title} data", transform=ax.transAxes, ha='center')
            return
        
        # Using scatter to represent the grid cells (rectangles)
        # Since we don't have a continuous surface, scatter with large markers or imshow-like approach?
        # A better way for "grid" is to use hexbin or just scatter if it's sparse. 
        # Given requirements "build a stack", let's treat cells as points in (time, price) space.
        sc = ax.scatter(cells['grid_t'], cells['grid_p'], c=cells['color'].apply(lambda x: x[:3]), s=50)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    if buy_cells is not None:
        plot_cells(ax1, buy_cells, "BUY Trades Impact")
    if sell_cells is not None:
        plot_cells(ax2, sell_cells, "SELL Trades Impact")

    # Set X-axis ticks to whole hours
    all_times = []
    if buy_cells is not None: all_times.extend(buy_cells['grid_t'].tolist())
    if sell_cells is not None: all_times.extend(sell_cells['grid_t'].tolist())

    if all_times:
        start_time = min(all_times)
        end_time = max(all_times)
        # Find first whole hour >= start_time
        first_hour = start_time.replace(minute=0, second=0, microsecond=0)
        if first_hour < start_time:
            first_hour += timedelta(hours=1)
        
        current = first_hour
        hour_ticks = []
        while current <= end_time:
            hour_ticks.append(current)
            current += timedelta(hours=1)
        
        plt.xticks(hour_ticks, [h.strftime('%H:%M') for h in hour_ticks], rotation=45)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Result saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time_step", type=int, default=60)
    parser.add_argument("--price_step", type=float, default=1.0)
    parser.add_argument("--percentile_grid", type=int, default=10)
    args = parser.parse_args()

    process_data(args.input, args.output, args.time_step, args.price_step, args.percentile_grid)
