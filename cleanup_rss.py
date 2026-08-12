import json
import csv
from pathlib import Path

results_dir = Path("results")

for raw_csv in results_dir.glob("*_raw.csv"):
    print(f"Processing {raw_csv}")
    rows = []
    with raw_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn for fn in reader.fieldnames if fn != "max_rss_bytes"]
        for row in reader:
            if "max_rss_bytes" in row:
                del row["max_rss_bytes"]
            rows.append(row)
    
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

for summary_json in results_dir.glob("*_summary.json"):
    print(f"Processing {summary_json}")
    with summary_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    for item in data:
        if "max_rss_bytes" in item:
            del item["max_rss_bytes"]
            
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

print("Memory cleanup complete.")
