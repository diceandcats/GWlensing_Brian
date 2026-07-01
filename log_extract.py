# pylint: skip-file

import csv
import glob
import math
import os
import re

cluster_re = re.compile(r"^Cluster [0-5] DE best chi\^2:")
fx_re = re.compile(r"^differential_evolution step \d+: f\(x\)=\s*([0-9]*\.[0-9]+|[0-9]+)")


rows = []
os.chdir("/home/dices/Research/GWlensing_Brian/outputs/with_z")
for fname in sorted(glob.glob("*.out")):
    vals = []
    lines = open(fname, "r", encoding="utf-8", errors="ignore").read().splitlines()
    for i, line in enumerate(lines):
        if cluster_re.match(line):
            m = fx_re.match(lines[i - 1].strip())
            if m:
                vals.append(float(m.group(1)))
    second_smallest = sorted(vals)[1]
    third_smallest = sorted(vals)[2]
    fourth_smallest = sorted(vals)[3]
    fifth_smallest = sorted(vals)[4]
    sixth_smallest = sorted(vals)[5]
    file_number = re.search(r"\d+", fname).group(0)
    rows.append((file_number, second_smallest, third_smallest, fourth_smallest, fifth_smallest, sixth_smallest))

with open("all_cluster_candidates.csv", "w", newline="") as f:
    w = csv.writer(f)
    # sort the rows by file number
    rows.sort(key=lambda x: int(x[0]))
    w.writerow(["file_number", "second_smallest_chi2", "third_smallest_chi2", "fourth_smallest_chi2", "fifth_smallest_chi2", "sixth_smallest_chi2"])
    w.writerows(rows)


base_dir = "/home/dices/Research/GWlensing_Brian"
src_pos_path = os.path.join(base_dir, "src_pos_tidy_xyz.csv")
cluster_candidates_path = os.path.join(os.getcwd(), "all_cluster_candidates.csv")
candidate_chi2_fields = [
    "second_smallest_chi2",
    "third_smallest_chi2",
    "fourth_smallest_chi2",
    "fifth_smallest_chi2",
    "sixth_smallest_chi2",
]


def to_float(value):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def logsumexp(values):
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


with open(cluster_candidates_path, newline="") as f:
    candidate_rows = list(csv.DictReader(f))

with open(src_pos_path, newline="") as f:
    reader = csv.DictReader(f)
    src_fieldnames = list(reader.fieldnames or [])
    src_rows = list(reader)

if "posterior_odds" not in src_fieldnames:
    src_fieldnames.append("posterior_odds")

chi_sq_limit = 50
for i, src_row in enumerate(src_rows):
    posterior_odds = "NaN"
    chi_sq = to_float(src_row.get("chi_sq"))

    if chi_sq is not None and chi_sq < chi_sq_limit and i < len(candidate_rows):
        candidate_chi2_values = [
            to_float(candidate_rows[i].get(field))
            for field in candidate_chi2_fields
        ]
        candidate_chi2_values = [value for value in candidate_chi2_values if value is not None]

        if candidate_chi2_values:
            log_denominator = logsumexp([-value for value in candidate_chi2_values])
            posterior_odds = str(-chi_sq - log_denominator)

    src_row["posterior_odds"] = posterior_odds

with open(src_pos_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=src_fieldnames)
    writer.writeheader()
    writer.writerows(src_rows)
