# pylint: skip-file

import re, glob, csv, os

cluster_re = re.compile(r'^Cluster [0-5] DE best chi\^2:')
fx_re = re.compile(r'^differential_evolution step \d+: f\(x\)=\s*([0-9]*\.[0-9]+|[0-9]+)')

rows = []
os.chdir("/home/dices/Research/GWlensing_Brian/with_z")
for fname in sorted(glob.glob("*.out")):
    vals = []
    lines = open(fname, "r", encoding="utf-8", errors="ignore").read().splitlines()
    for i, line in enumerate(lines):
        if cluster_re.match(line):
            m = fx_re.match(lines[i - 1].strip())
            if m:
                vals.append(float(m.group(1)))
    second_smallest = sorted(vals)[1]
    file_number = re.search(r'\d+', fname).group(0)
    rows.append((file_number, second_smallest))
print(rows)

with open("second_smallest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file_number", "second_smallest_chi2"])
    w.writerows(rows)