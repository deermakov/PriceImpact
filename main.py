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
            prices = group['price'].values
            # Store raw prices for reporting
            raw_prices = list(prices)
            if len(prices) > 1:
                diffs = np.diff(prices)
                val = np.mean(diffs)
                # If it's a SELL side, invert the sign of the mean difference
                if side_name == 'SELL':
                    val = -val
                diff_series = list(diffs)
            else:
                val = 0.0 
                diff_series = []
            
            cells.append({
                'grid_t': gt, 
                'grid_p': gp, 
                'value': val, 
                'raw_prices': raw_prices,
                'diff_series': diff_series
            })
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
        
        # Store bins for distribution reporting
        cells_df['_bins'] = None # Placeholder

        # 7. Assign color
        cmap = plt.get_cmap(palette)
        num_colors = len(bins) - 1
        cells_df['color'] = cells_df['quantile_idx'].apply(lambda x: cmap(x / (num_colors if num_colors > 0 else 1)))

        return cells_df, bins

    buy_cells_data = analyze_side('BUY', 'viridis')
    sell_cells_data = analyze_side('SELL', 'inferno')

    buy_cells = buy_cells_data[0] if buy_cells_data else None
    buy_bins = buy_cells_data[1] if buy_cells_data else None
    sell_cells = sell_cells_data[0] if sell_cells_data else None
    sell_bins = sell_cells_data[1] if sell_cells_data else None

    # 8. Report to text file
    report_file = output_file.replace('.png', '_report.txt')
    with open(report_file, 'w') as f:
        f.write("--- PRICE IMPACT REPORT ---\n\n")
        
        f.write("1. CELL DETAILS\n")
        for side, cells in [('BUY', buy_cells), ('SELL', sell_cells)]:
            f.write(f"\n[{side} SIDE]\n")
            if cells is None or cells.empty:
                f.write("No data available.\n")
                continue
            
            current_bins = buy_bins if side == 'BUY' else sell_bins
            
            for _, row in cells.iterrows():
                # Calculate boundaries exactly as they are in the grid
                t_start = row['grid_t']
                t_end = t_start + pd.Timedelta(seconds=time_step_sec)
                p_start = row['grid_p']
                p_end = p_start + price_step
                
                f.write(f"Cell: Time [{t_start} to {t_end}], Price [{p_start} to {p_end}]\n")
                f.write(f"  Prices: {row['raw_prices']}\n")
                f.write(f"  Diff Series: {row['diff_series']}\n")
                f.write(f"  Value: {row['value']:.6f}\n")
                # Find quantile index/range
                q_idx = row['quantile_idx']
                if current_bins is not None and 0 <= q_idx < len(current_bins) - 1:
                    f.write(f"  Quantile range: [{current_bins[q_idx]:.6f}, {current_bins[q_idx+1]:.6f}]\n")
                f.write("-" * 20 + "\n")

        f.write("\n2. TOTAL BUY DISTRIBUTION (Quantiles)\n")
        if buy_bins is not None:
            for i in range(len(buy_bins) - 1):
                f.write(f"  Q{i}: [{buy_bins[i]:.6f}, {buy_bins[i+1]:.6f}]\n")
        else:
            f.write("No data.\n")

        f.write("\n3. TOTAL SELL DISTRIBUTION (Quantiles)\n")
        if sell_bins is not None:
            for i in range(len(sell_bins) - 1):
                f.write(f"  Q{i}: [{sell_bins[i]:.6f}, {sell_bins[i+1]:.6f}]\n")
        else:
            f.write("No data.\n")

    print(f"Report saved to: {report_file}")

    # Plotting
    print("Plotting...")
    
    # Parameters for cell size in inches to keep them visually consistent
    CELL_WIDTH_INCHES = 0.1
    CELL_HEIGHT_INCHES = 0.1

    all_times = []
    if buy_cells is not None: all_times.extend(buy_cells['grid_t'].tolist())
    if sell_cells is not None: all_times.extend(sell_cells['grid_t'].tolist())

    if not all_times:
        print("No data to plot.")
        return

    start_time = min(all_times)
    end_time = max(all_times)

    # Calculate dynamic figsize
    duration_seconds = (end_time - start_time).total_seconds()
    num_cells_x = max(1, int(np.ceil(duration_seconds / time_step_sec)))
    total_width = num_cells_x * CELL_WIDTH_INCHES

    all_prices = []
    if buy_cells is not None: all_prices.extend(buy_cells['grid_p'].tolist())
    if sell_cells is not None: all_prices.extend(sell_cells['grid_p'].tolist())
    
    price_min = min(all_prices) if all_prices else 0
    price_max = max(all_prices) if all_prices else price_step
    price_range = max(price_step, price_max - price_min)
    
    num_cells_y = max(1, int(np.ceil(price_range / price_step)))
    # Height accounts for 2 plots + margins/labels
    total_height = (num_cells_y * CELL_HEIGHT_INCHES) * 2 + 3

    # Safety bounds to prevent extreme figsize
    total_width = max(10, min(total_width, 500))
    total_height = max(6, min(total_height, 500))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(total_width, total_height), sharex=True)

    def plot_cells(ax, cells, title):
        if cells is None or cells.empty:
            ax.text(0.5, 0.5, f"No {title} data", transform=ax.transAxes, ha='center')
            return
        
        for _, row in cells.iterrows():
            rect = plt.Rectangle(
                (row['grid_t'] - pd.Timedelta(seconds=time_step_sec/2), row['grid_p'] - price_step/2),
                pd.Timedelta(seconds=time_step_sec), 
                price_step,
                facecolor=row['color'],
                edgecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)

        ax.set_xlim(start_time - pd.Timedelta(seconds=time_step_sec), end_time + pd.Timedelta(seconds=time_step_sec))
        ax.set_ylim(price_min - price_step, price_max + price_step)
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, which='both', linestyle='-', alpha=0.7)

    if buy_cells is not None:
        plot_cells(ax1, buy_cells, "BUY Trades Impact")
        ax1_right = ax1.twinx()
        ax1_right.set_ylim(ax1.get_ylim())
        ax1_right.yaxis.set_ticks_position('right')

    if sell_cells is not None:
        plot_cells(ax2, sell_cells, "SELL Trades Impact")
        ax2_right = ax2.twinx()
        ax2_right.set_ylim(ax2.get_ylim())
        ax2_right.yaxis.set_ticks_position('right')

    if all_times:
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
            start_minute = (start_time.minute // 10) * 10
            current_fine = start_time.replace(minute=start_minute, second=0, microsecond=0)
            if current_fine < start_time:
                current_fine += timedelta(minutes=10)
            
            while current_fine <= end_time:
                fine_ticks.append(current_fine)
                current_fine += timedelta(minutes=10)
            
            all_ticks = sorted(list(set(hour_ticks + fine_ticks)))
        else:
            all_ticks = hour_ticks

        if all_ticks:
            ax2.set_xticks(all_ticks)
            ax2.set_xticklabels([t.strftime('%H:%M') for t in all_ticks], rotation=45, fontsize=14)
            ax1.set_xticks(all_ticks)
            ax1.set_xticklabels([t.strftime('%H:%M') for t in all_ticks], rotation=45, fontsize=14)
            # Ensure y-axis labels are also large
            ax1.tick_params(axis='y', labelsize=14)
            ax2.tick_params(axis='y', labelsize=14)

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
