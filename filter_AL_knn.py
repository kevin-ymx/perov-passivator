"""Remove rows from AL_knn_results.csv whose ref_cid appears in strong_binder_knn_results.csv."""
import csv

# Step 1: collect ref_cids to exclude
exclude_cids = set()
with open("strong_binder_knn_results.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        exclude_cids.add(row["ref_cid"].strip())

print("Unique ref_cids to exclude:", len(exclude_cids))

# Step 2: read AL_knn_results.csv and filter
kept_rows = []
removed = 0
with open("AL_knn_results.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        if row["ref_cid"].strip() in exclude_cids:
            removed += 1
        else:
            kept_rows.append(row)

# Step 3: write filtered result back
with open("AL_knn_results.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kept_rows)

print("Removed", removed, "rows, kept", len(kept_rows), "rows")
