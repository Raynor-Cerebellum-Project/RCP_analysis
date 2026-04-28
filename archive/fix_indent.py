
from pathlib import Path

path = Path(r"e:\NHP_Cerebellum_Project_2025_RCP_Analysis_Git\RCP_analysis\scripts\analyze_stim_reach.py")
lines = path.read_text("utf-8").splitlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "if cam0.size:" in line and "interp_nans" not in line:
        # Avoid matching other occurrences if any
        # The target line is strict: "        if cam0.size:" (8 spaces)
        if line.strip() == "if cam0.size:":
            start_idx = i
            break

for i, line in enumerate(lines):
    if "if __name__ ==" in line:
        end_idx = i
        break

if start_idx == -1:
    print("Could not find start line")
    exit(1)

if end_idx == -1:
    end_idx = len(lines)

print(f"Indenting lines {start_idx} to {end_idx}")

new_lines = []
for i, line in enumerate(lines):
    if i >= start_idx and i < end_idx:
        # Add 4 spaces if line is not empty
        if line.strip():
            new_lines.append("    " + line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

path.write_text("\n".join(new_lines), "utf-8")
print("Done.")
