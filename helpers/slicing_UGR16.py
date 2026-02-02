import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, 'data')

# Set output file name
output_filename = os.path.join(data_dir, "1ugr_experiment_data_2016-07-29_0200-0400.csv")
input_filename = os.path.join(data_dir, "july.week5.csv.uniqblacklistremoved")

# Set data window
target_date = "2016-07-29"
start_time = "02:00:00"
end_time = "04:00:00"

if os.path.exists(input_filename):
    total_bytes = os.path.getsize(input_filename)
    print(f"Input file size: {total_bytes / (1024**3):.2f} GB")
else:
    print("Error: Input file not found")
    exit()

print(f"Reading from: {input_filename}")
print(f"Saving to:    {output_filename}")
print("\n" + "-" * 50 + "\n")

header_names = [
    "Timestamp", "Duration", "SrcIP", "DstIP", "SrcPort", "DstPort", 
    "Proto", "Flag", "ForwardingStatus", "ToS", "Packets", "Bytes", "Label"
]

chunksize = 1_000_000
rows_saved = 0
rows_processed = 0
first_chunk = True

try:
    with open(input_filename, 'r') as f:
        try:
            reader = pd.read_csv(f, names=header_names, chunksize=chunksize, on_bad_lines='skip')

        except TypeError:
            reader = pd.read_csv(f, names=header_names, chunksize=chunksize, error_bad_lines=False)

        for chunk in reader:
            chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            filtered_data = chunk.loc[chunk['Timestamp'].between(f"{target_date} {start_time}", f"{target_date} {end_time}", inclusive='left')]

            if not filtered_data.empty:
                mode = 'w' if first_chunk else 'a'
                filtered_data.to_csv(output_filename, mode=mode, header=False, index=False, date_format='%Y-%m-%d %H:%M:%S')
                rows_saved += len(filtered_data)
                first_chunk = False
            
            # Show progress
            rows_processed += len(chunk)
            current_bytes = f.tell()
            percentage_complete = (current_bytes / total_bytes) * 100
            
            print(f"Progress: {percentage_complete:5.2f}% | Processed {rows_processed:,} lines | Saved {rows_saved:,} relevant rows", end='\r')
            
            # If input data is chronological, stop early after finding date that is later than window end
            valid_ts = chunk['Timestamp'].dropna()
            if not valid_ts.empty and valid_ts.iloc[-1] > pd.Timestamp(f"{target_date} {end_time}"):
                print("\nPassed target window, stopping search")
                break

    print(f"\n\nSUCCES: Saved {rows_saved} rows to '{output_filename}'")

except Exception as e:
    print(f"\nERROR: An error occurred: {e}")