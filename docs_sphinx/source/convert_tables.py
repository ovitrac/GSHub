#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_tables.py

Convert Markdown pipe tables (GitHub style) to MyST-compatible grid tables (Sphinx).
Backup of the input file is automatically created as *.bak.md.

Usage:
    ./convert_tables.py README.md

Author: Generative Simulation | olivier.vitrac@gmail.com
"""

import re
import sys
from pathlib import Path
from shutil import copyfile

def parse_tables(md_lines):
    tables = []
    current = []
    inside_table = False

    for line in md_lines:
        if re.match(r'\s*\|', line):
            current.append(line.rstrip())
            inside_table = True
        else:
            if inside_table and len(current) >= 2:
                tables.append(current)
            current = []
            inside_table = False
    if inside_table and len(current) >= 2:
        tables.append(current)
    return tables

def pipe_to_grid(table_lines):
    rows = [re.split(r'\s*\|\s*', line.strip('|')) for line in table_lines if '|' in line]
    ncols = len(rows[0])
    col_widths = [max(len(cell) for cell in col) for col in zip(*rows)]

    def sep(char='-', junction='+', header=False):
        return junction + junction.join(
            (('=' if header else char) * (w + 2)) for w in col_widths
        ) + junction

    def format_row(cells):
        return '| ' + ' | '.join(f"{cell:<{w}}" for cell, w in zip(cells, col_widths)) + ' |'

    output = [sep('-')]
    output.append(format_row(rows[0]))
    output.append(sep('=', header=True))
    for row in rows[2:]:
        output.append(format_row(row))
        output.append(sep('-'))
    return output

def replace_tables(content):
    lines = content.splitlines()
    tables = parse_tables(lines)
    if not tables:
        return content  # No change

    out_lines = []
    i = 0
    while i < len(lines):
        if any(lines[i].strip() == row.strip() for table in tables for row in table):
            for table in tables:
                if lines[i:i+len(table)] == table:
                    out_lines.extend(pipe_to_grid(table))
                    i += len(table)
                    break
        else:
            out_lines.append(lines[i])
            i += 1
    return '\n'.join(out_lines)

def main(mdfile):
    path = Path(mdfile)
    if not path.exists():
        print(f"❌ File not found: {mdfile}")
        sys.exit(1)

    backup = path.with_suffix('.bak.md')
    copyfile(path, backup)
    original = path.read_text(encoding='utf-8')
    converted = replace_tables(original)
    path.write_text(converted, encoding='utf-8')
    print(f"✔️ Converted pipe tables in {mdfile} → grid format (backup: {backup.name})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./convert_tables.py README.md")
    else:
        main(sys.argv[1])

