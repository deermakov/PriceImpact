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
    try:
        date_str = os.path.basename(input_file).split('.')[0]
        base_date = datetime.strptime(date_str, '%Y.%m.%d')
    except:
        base_date = datetime(2026, 8, 10)

    def parse_time(row):
        t = datetime.strptime(row['TRADETIME'], '%H:%M:%S')
        # TRADETIME_MSEC is treated as microseconds
        return base_date + timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=int(row['TRADETIME_MSEC']))

    df['timestamp'] = df.apply(parse_time, axis=1)
    df['price'] = pd.to_numeric(df['PRICE'], errors='coerce')
    df['BUYSELL'] = df['BUYSELL'].str.upper()

    # Define grid parameters
    def get_grid_coords(row):
        t_seconds = row['timestamp'].timestamp()
        grid_t = (t_seconds // time_step_sec) * time_step_sec
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
            prices = group['price'].sort_values().values
            if len(prices) > 1:
                diffs = np.diff(prices)
                val = np.mean(diffs)
            else:
                val = 0.0 
            
            cells.append({'grid_t': gt, 'grid_p': gp, 'value': val})
            values.append(val)

        if not cells:
            return None

        cells_df = pd.DataFrame(cells)
        vals = np.array(values)

        # 5. Quantiles
        num_fragments = 100 // percentile_grid_size
        quantiles = np.linspace(0, 100, num_fragments + 1)
        bins = np.percentile(vals, quantiles)
        bins = np.unique(bins)
        
        cells_df['quantile_idx'] = np.digitize(cells_df['value'], bins) - 1
        cells_df['quantile_idx'] = cells_df['quantile_idx'].clip(0, len(bins) - 2)

        # 7. Assign color
        cmap = plt.get_cmap(palette)
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
        
        for _, row in cells.iterrows():
            # Matplotlib Rectangle expects numerical values for width/height when working with datetime axes
            # We convert the time offset to a Timedelta and use it directly, 
            # but ensure the width is passed as a Timedelta object so matplotlib can handle it.
            rect = plt.Rectangle(
                (row['grid_t'] - pd.Timedelta(seconds=time_step_sec/2), row['grid_p'] - price_step/2),
                pd.Timedelta(seconds=time_step_sec), 
                price_step,
                facecolor=row['color'],
                edgecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)

        ax.set_xlim(cells['grid_t'].min() - pd.Timedelta(seconds=time_step_sec), cells['grid_t'].max() + pd.Timedelta(seconds=time_step_sec))
        ax.set_ylim(cells['grid_p'].min() - price_step, cells['grid_p'].max() + price_step)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='-', alpha=0.7)

    if buy_cells is not None:
        plot_cells(ax1, buy_cells, "BUY Trades Impact")
    if sell_cells is not None:
        plot_cells(ax2, sell_cells, "SELL Trades Impact")

    all_times = []
    if buy_cells is not None: all_times.extend(buy_cells['grid_t'].tolist())
    if sell_cells is not None: all_times.extend(sell_cells['grid_t'].tolist())

    if all_times:
        start_time = min(all_times)
        end_time = max(all_times)
        
        # Logic for x-axis ticks based on time_step_sec
        hour_ticks = []
        current = start_time.replace(minute=0, second=0, microsecond=0)
        if current < start_time:
            current += timedelta(hours=1)
            
        while current <= end_time:
            hour_ticks.append(current)
            current += timedelta(hours=1)

        # Add finer ticks if time_step_sec < 3600 (e.g., every 10 mins)
        if time_step_sec < 3600:
            fine_ticks = []
            # Start from the first full 10-min interval after or at start_time
            start_minute = (start_time.minute // 10) * 10
            current_fine = start_time.replace(minute=start_minute, second=0, microsecond=0)
            if current_fine < start_time:
                current_fine += timedelta(minutes=10)
            
            while current_fine <= end_time:
                fine_ticks.append(current_fine)
                current_fine += timedelta(minutes=10)
            
            # Combine and sort unique ticks
            all_ticks = sorted(list(set(hour_ticks + fine_ticks)))
        else:
            all_ticks = hour_ticks

        if all_ticks:
            plt.xticks(all_ticks, [t.strftime('%H:%M') for t in all_ticks], rotation=45)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Result saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--time_step", type=int, default=60)
    parser.add_argument("--price_step", type=float, default=1.0)
    parser.add_argument("--percentile_grid", type=int, default=10)
    args = parser.parse_args()

    # Override with environment variables if they exist
    input_file = os.getenv('INPUT_FILE', args.input)
    output_file = os.getenv('OUTPUT_FILE', args.output)
    time_step = int(os.getenv('TIME_STEP_SEC', args.time_step))
    price_step = float(os.getenv('PRICE_STEP', args.price_step))
    percentile_grid = int(os.getenv('PERCENTILE_GRID_SIZE', args.percentile_grid))

    if not input_file or not output_file:
        parser.error("Both --input/INPUT_FILE and --output/OUTPUT_FILE must be provided.")

    process_data(input_file, output_file, time_step, price_step, percentile_grid)
