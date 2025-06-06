#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_doctoc.py

Fix doctoc-generated GitHub-style TOC anchors for Sphinx/MyST
- Finds `doctoc` anchor links like `[#-1-main-components]`
- Rewrites them to match Sphinx-style or JupyterBook anchor IDs

Usage:
    Run in the same directory as README.md

Generative Simulation | olivier.vitrac@gmail.com
"""

import re
from pathlib import Path
from shutil import copyfile

def fix_anchor(label: str) -> str:
    # Remove leading dashes and numbers (e.g., "-5--title")
    label = re.sub(r'^[-\d\s]*', '', label)
    # Replace remaining dashes with spaces (normalize)
    label = label.replace('-', ' ')
    # Slugify: lowercase, trim, replace spaces with dashes
    slug = re.sub(r'\s+', '-', label.strip().lower())
    return f'#{slug}'

def fix_doctoc_links(md_path: Path):
    backup_path = md_path.with_suffix(".bak.md")
    copyfile(md_path, backup_path)
    content = md_path.read_text(encoding='utf-8')

    # Fix links like (#-5--Sinusoidal-Encoding)
    pattern = re.compile(r'\(#([^)]+)\)')
    content = pattern.sub(lambda m: f'({fix_anchor(m.group(1))})', content)

    md_path.write_text(content, encoding='utf-8')
    print(f"✔️ Fixed TOC links in {md_path.name} (backup saved as {backup_path.name})")

if __name__ == "__main__":
    readme = Path("README.md")
    if readme.exists():
        fix_doctoc_links(readme)
    else:
        print("❌ README.md not found.")
