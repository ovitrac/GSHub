#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_readme_equations.py

Fix math in README.md for Sphinx + MyST compatibility.
- Converts $$...$$ and [ ... ] to ```{math} ... ```
- Escapes underscores (_) outside code/math/links/images
- Preserves image syntax ![alt](path)
- Creates a backup of the original README.md

Usage:
    Run in the same directory as README.md

Generative Simulation | olivier.vitrac@gmail.com
"""

import re
from pathlib import Path
import shutil

def backup_file(path):
    backup = path.with_name(path.stem + '_backup.md')
    shutil.copy(path, backup)
    print(f"📦 Backup created: {backup.name}")

def convert_math_blocks(text):
    # Temporarily protect image references
    image_refs = re.findall(r'!\[[^\]]*\]\([^)]+\)', text)
    placeholders = {f"__IMG{i}__": ref for i, ref in enumerate(image_refs)}
    for placeholder, original in placeholders.items():
        text = text.replace(original, placeholder)

    # Convert [ ... ] to ```{math} ... ``` if not inside a line (must be standalone)
    text = re.sub(r'(?m)^\[\s*(.*?)\s*\]$', r'```{math}\n\1\n```', text)

    # Convert $$ ... $$ to ```{math} ... ``` (block math only)
    text = re.sub(r'\$\$\s*(.*?)\s*\$\$', r'```{math}\n\1\n```', text, flags=re.DOTALL)

    # Restore image references
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder, original)

    return text

def escape_underscores(text):
    # Escape _ in plain text lines, avoid code, math, links/images
    lines = text.splitlines()
    escaped = []
    inside_block = False

    for line in lines:
        if line.strip().startswith("```"):
            inside_block = not inside_block
            escaped.append(line)
            continue

        if inside_block or line.strip().startswith("    "):
            escaped.append(line)
            continue

        # Skip Markdown image or link lines
        if re.match(r'!\[.*\]\(.*\)', line) or re.match(r'\[.*\]\(.*\)', line):
            escaped.append(line)
            continue

        escaped_line = re.sub(r'(?<!\\)_', r'\_', line)
        escaped.append(escaped_line)

    return "\n".join(escaped)

def fix_readme(input_path='README.md', output_path='README.md'):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"❌ Input file {input_path} not found.")

    backup_file(input_file)

    content = input_file.read_text(encoding='utf-8')
    content = convert_math_blocks(content)
    content = escape_underscores(content)

    output_file.write_text(content, encoding='utf-8')
    print(f"✅ Sphinx-compatible file written: {output_file.name}")

if __name__ == "__main__":
    fix_readme()
