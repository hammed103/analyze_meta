#!/usr/bin/env python3

# Read the file
with open('summaary.txt', 'r') as f:
    lines = f.readlines()

# Filter out lines containing "Key Message/Slogan" or "Overall Theme"
cleaned_lines = []
for line in lines:
    if "Key Message/Slogan" not in line and "Overall Theme" not in line:
        cleaned_lines.append(line)

# Write back to file
with open('summaary.txt', 'w') as f:
    f.writelines(cleaned_lines)

print("Cleaned summaary.txt - removed all 'Key Message/Slogan' and 'Overall Theme' lines")
