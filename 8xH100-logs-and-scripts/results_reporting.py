import re
import os
import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import ttest_ind

baseline_name = "baseline"
logs_folder = "./logs"

# ======== Read Log Files, Infer Group From Folder ========

# Retrieve all log files in the logs folder.
# Assign them to groups based on their subfolder name.
log_inputs = []

for subfolder in os.listdir(logs_folder):
    subfolder_path = os.path.join(logs_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
    for log_file in os.listdir(subfolder_path):
        if log_file.endswith('.txt'):
            log_inputs.append(
                (
                    subfolder,  # Use the subfolder name as the group name.
                    log_file[:-4],  # Use the filename as the run name, stripping off the .txt.
                    f"{logs_folder}/{subfolder}/{log_file}",
                )
            )

#print(log_inputs)

# ======== Parse Log Files ========

# Regex to capture the different parts of the log line
line_regex = re.compile(
    r"step:(\d+)/(\d+)"                     # Group 1 & 2: step/total_steps
    r"(?: val_loss:([\d\.]+))?"             # Group 3: val_loss (optional)
    r"(?: train_time:(\d+)ms)?"             # Group 4: train_time (optional)
    r"(?: step_avg:([\d\.]+)ms)?"           # Group 5: step_avg (optional)
)

all_dfs = [] # To store DataFrames from each file

# Unpack group, run_name, and path
for group_name, run_name, file_path in log_inputs:
    data = []
    parsing_started = False

    # Retrieve file content from URL or local file
    try:
        if file_path.startswith("https://"):
            response = requests.get(file_path)
            response.raise_for_status()
            file_content = response.text
        elif not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}.")
            continue
        else:
            with open(file_path, 'r') as f:
                file_content = f.read()

        # Parse the content line by line
        for line in file_content.splitlines():
            line = line.strip()

            if not parsing_started:
                if line.startswith("step:0"):
                    parsing_started = True
                else:
                    continue

            match = line_regex.match(line)
            if match:
                groups = match.groups()
                # If we have val_loss but no train_time, this is a validation-only line
                # Update the previous entry with the val_loss
                if groups[2] and not groups[3]:
                    if data:
                        data[-1]['val_loss'] = float(groups[2])
                else:
                    data.append({
                        "step": int(groups[0]),
                        "total_steps": int(groups[1]),
                        "val_loss": float(groups[2]) if groups[2] else np.nan,
                        "train_time_ms": int(groups[3]) if groups[3] else np.nan,
                        "step_avg_ms": float(groups[4]) if groups[4] else np.nan,
                    })

        if not data:
            print(f"Warning: Log file {file_path} does not contain any training run data (likely crashed before logging started). Skipping.")
            continue

        df = pd.DataFrame(data)
        df = df.groupby('step').last().reset_index()

        # Store both Group and Run Name
        df['group_name'] = group_name
        df['run_name'] = run_name
        all_dfs.append(df)

    except KeyError as e:
        if 'step' in str(e):
            print(f"Warning: Log file {file_path} does not contain any training run data (likely crashed before logging started). Skipping.")
        else:
            print(f"Warning: Error processing {file_path}. Error: {e}. Skipping.")
    except Exception as e:
        print(f"Warning: Error processing {file_path}. Error: {e}. Skipping.")

if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['train_time_s'] = combined_df['train_time_ms'] / 1000
else:
    print("No data frames were created.")
    combined_df = pd.DataFrame()

# ======== Generate Summary Table ========

if not combined_df.empty:
    # 1. Get the final data point for each unique run
    # We sort by step to ensure we get the absolute last logged event for each run
    final_points_df = combined_df.sort_values('step').groupby('run_name').last().reset_index()

    # 2. Group by 'group_name' and calculate count/mean/std for time and loss
    summary_df = final_points_df.groupby('group_name').agg(
        Runs=('run_name', 'nunique'),
        Time_mu=('train_time_s', 'mean'),
        Time_std=('train_time_s', 'std'),
        Loss_mu=('val_loss', 'mean'),
        Loss_std=('val_loss', 'std'),
    )

    # 3. Add Time +/- and Loss +/- as *absolute* differences vs baseline (not %)
    if baseline_name in summary_df.index:
        baseline_time_mu = summary_df.loc[baseline_name, 'Time_mu']
        baseline_loss_mu = summary_df.loc[baseline_name, 'Loss_mu']

        summary_df['Time_diff'] = summary_df['Time_mu'] - baseline_time_mu
        summary_df['Loss_diff'] = summary_df['Loss_mu'] - baseline_loss_mu
    else:
        print(f"Warning: '{baseline_name}' group not found. +/- differences will be NaN.")
        summary_df['Time_diff'] = float('nan')
        summary_df['Loss_diff'] = float('nan')

    # 4. Compute p-values for loss:
    #    p = scipy.stats.ttest_1samp(losses, 3.28, alternative='less').pvalue
    p_values = {}
    for group_name, group_df in final_points_df.groupby('group_name'):
        losses = group_df['val_loss'].dropna().astype(float).values
        if len(losses) > 0:
            p = scipy.stats.ttest_1samp(losses, 3.28, alternative='less').pvalue
        else:
            p = np.nan
        p_values[group_name] = p

    summary_df['p'] = summary_df.index.map(p_values)


    # p-values for time
    time_p_values = {}

    if baseline_name in final_points_df['group_name'].values:
        baseline_times = (
            final_points_df[final_points_df['group_name'] == baseline_name]
            ['train_time_s']
            .dropna()
            .values
        )

        for group_name, group_df in final_points_df.groupby('group_name'):
            times = group_df['train_time_s'].dropna().values

            if group_name == baseline_name or len(times) == 0:
                time_p_values[group_name] = np.nan
                continue

            p = ttest_ind(
                times,
                baseline_times,
                alternative="less",   # faster than baseline
                equal_var=False       # Welch
            ).pvalue

            time_p_values[group_name] = p

        summary_df['Time_p'] = summary_df.index.map(time_p_values)    
    else:
        print("Warning: baseline not found for time p-values.")
        # Always create Time_p column, set to NaN if baseline is missing
        summary_df['Time_p'] = np.nan



    # 5. Rename columns to use μ and σ, and +/- labels as requested
    summary_df = summary_df[[
        'Runs',
        'Time_mu', 'Time_std', 'Time_diff', 'Time_p',
        'Loss_mu', 'Loss_std', 'Loss_diff',
        'p'
    ]]

    summary_df.rename(columns={
        'Time_mu':  'Time μ',
        'Time_std': 'Time σ',
        'Time_diff': 'Time +/-',
        'Time_p': 'Time p',
        'Loss_mu':  'Loss μ',
        'Loss_std': 'Loss σ',
        'Loss_diff': 'Loss +/-',
    }, inplace=True)

    # Remove the index name for cleaner display
    summary_df.index.name = None

    # 6. Display the table formatted to 4 decimal places
    pd.options.display.float_format = "{:,.4f}".format
    # Ensure all columns are displayed
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(summary_df)

    # Save styled summary table as HTML
    styled_table = (
        summary_df.style
            .format({
                'Runs': '{:.0f}',  # Format 'Runs' as integer
                'Time μ': '{:,.2f}',
                'Time σ': '{:,.2f}',
                'Time +/-': '{:,.2f}',
                'Time p': '{:,.4f}',
                'Loss μ': '{:,.4f}',
                'Loss σ': '{:,.4f}',
                'Loss +/-': '{:,.4f}',
                'p': '{:,.4f}'
            })
            .background_gradient(cmap='coolwarm', subset=['Loss +/-'])
            .background_gradient(cmap='coolwarm', subset=['Time +/-'])
    )

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    html_path = f"plots/summary_table_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"
    with open(html_path, 'w') as f:
        f.write(styled_table.to_html())
    print(f"Saved styled table to {html_path}")

# ======================================================
#      Display Data Points as Array Literals
# ======================================================

group_loss_arrays = {}
group_time_arrays = {}

if not combined_df.empty:
    # Keep only rows that actually have a val_loss (for the loss arrays)
    valid_rows = combined_df.dropna(subset=["val_loss"])

    # For each group → for each run → get the final logged val_loss and train_time_s
    for group_name, group_df in valid_rows.groupby("group_name"):
        final_rows = (
            group_df.sort_values("step")   # ensure increasing step order
                   .groupby("run_name")    # isolate each run
                   .tail(1)                # last row per run
        )

        final_losses = final_rows["val_loss"].astype(float).tolist()
        final_times  = final_rows["train_time_s"].astype(float).tolist()

        group_loss_arrays[group_name] = final_losses
        group_time_arrays[group_name] = final_times

print("\n=== Final Validation Loss & Time Arrays Per Group ===\n")
for group_name in group_loss_arrays.keys():
    losses = group_loss_arrays[group_name]
    times  = group_time_arrays.get(group_name, [])

    loss_literal = "[" + ", ".join(f"{v:.4f}" for v in losses) + "]"
    time_literal = "[" + ", ".join(f"{t:.4f}" for t in times) + "]"

    print(f"{group_name}:")
    print(f"  losses = {loss_literal}")
    print(f"  times  = {time_literal}")
    print(f"  (n = {len(losses)})\n")

# ======================================================
#      Plot Full Training Curve - Val Loss vs. Time
# ======================================================

y_min = 3.2
y_max = 4.1
x_min = 0
x_max = 150

if not combined_df.empty:
    # 1. Filter for valid data points
    valid_data = combined_df.dropna(subset=['val_loss', 'step'])

    # 2. Group by 'group_name' and 'step'.
    # This averages the train_time and val_loss across all 'run_names' in that group.
    averaged_data = valid_data.groupby(['group_name', 'step'])[['train_time_s', 'val_loss']].mean().reset_index()

    unique_groups = averaged_data['group_name'].unique()
    colors = plt.get_cmap('tab10', len(unique_groups))

    plt.figure(figsize=(10, 6), dpi=300)

    for i, group_name in enumerate(unique_groups):

        # Select data for this specific group
        plot_data = averaged_data[averaged_data['group_name'] == group_name]

        plt.plot(
            plot_data['train_time_s'],
            plot_data['val_loss'],
            'o-',
            label=group_name,
            color=colors(i),
            markersize=4,
            linewidth=2
        )

    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)

    plt.xlabel('Average Train Time (s)')
    plt.ylabel('Average Validation Loss')
    plt.title('Validation Loss over Time\n(Averaged over Multiple Runs)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/val_loss_vs_time_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
else:
    print("DataFrame is empty, nothing to plot.")

# ======== Plot of Final Steps ========
# Validation loss vs. time for the final steps of the runs in each group.

y_min = 3.27
y_max = 3.3
x_min = 120
x_max = 127

if not combined_df.empty:
    # 1. Filter for valid data points
    valid_data = combined_df.dropna(subset=['val_loss', 'step'])

    # 2. Group by 'group_name' and 'step'.
    # This averages the train_time and val_loss across all 'run_names' in that group.
    averaged_data = valid_data.groupby(['group_name', 'step'])[['train_time_s', 'val_loss']].mean().reset_index()

    unique_groups = averaged_data['group_name'].unique()
    colors = plt.get_cmap('tab10', len(unique_groups))

    plt.figure(figsize=(10, 6))

    for i, group_name in enumerate(unique_groups):

        # Select data for this specific group
        plot_data = averaged_data[averaged_data['group_name'] == group_name]

        plt.plot(
            plot_data['train_time_s'],
            plot_data['val_loss'],
            'o-',
            label=group_name,
            color=colors(i),
            markersize=4,
            linewidth=2
        )

    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)

    plt.xlabel('Average Train Time (s)')
    plt.ylabel('Average Validation Loss')
    plt.title('Mean Validation Loss over Time (Grouped)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/val_loss_vs_time_final_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
else:
    print("DataFrame is empty, nothing to plot.")


# -----------------------------------------------------------
#      Required Reporting for Submission
# -----------------------------------------------------------

# ======== 12-10 Baseline ========
accs = np.array([3.2796, 3.2781, 3.2798, 3.2793])
times = np.array([132.5520, 132.8170, 132.7390, 132.7790])
p_value = scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue
print(f"p = {p_value:.4}")
print(f"Loss std, mean: {accs.std():.4f}, {accs.mean():.4f}")
print(f"Time std, mean: {times.std():.4f}, {times.mean():.4f}")
# p = 0.06323
# Loss std, mean:  0.0007, 3.2792
# Time std, mean:  0.1019, 132.7218

# ======== This Record ========
accs = np.array([3.2758, 3.2807, 3.2776, 3.2757, 3.2793, 3.2764, 3.2795, 3.2788, 3.2785, 3.2748])
times = np.array([131.2380, 131.2910, 131.1550, 131.1570, 131.2540, 131.3040, 131.1550, 131.1300, 131.1570, 131.2650])
p_value = scipy.stats.ttest_1samp(accs, 3.28, alternative='less').pvalue
print(f"p = {p_value:.4}")
print(f"Loss std, mean: {accs.std():.4f}, {accs.mean():.4f}")
print(f"Time std, mean: {times.std():.4f}, {times.mean():.4f}")
# p = 0.002439
# Loss std, mean:  0.0018, 3.2777
# Time std, mean:  0.0626, 131.2106

