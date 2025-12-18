#!/usr/bin/env python3
"""
Fix undefined variable references in the notebook.
"""

import json

# Load notebook
with open('corrected-moe-FIXED-V4.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

def get_cell_source(cell):
    """Get source as string from cell."""
    source = cell.get('source', [])
    if isinstance(source, list):
        return ''.join(source)
    return source

def set_cell_source(cell, text):
    """Set source from string to cell."""
    lines = text.split('\n')
    cell['source'] = [line + '\n' if i < len(lines) - 1 else line 
                      for i, line in enumerate(lines)]

cells = notebook['cells']

for i, cell in enumerate(cells):
    source = get_cell_source(cell)
    original = source
    
    # Fix train_dataset -> train_dataset_tokenized in MoETrainer instantiation
    if 'trainer = MoETrainer(' in source and 'train_dataset=train_dataset,' in source:
        source = source.replace(
            'train_dataset=train_dataset,',
            'train_dataset=train_dataset_tokenized,'
        )
        print(f"Fixed train_dataset reference in cell {i}")
    
    # Fix eval_dataset -> eval_dataset_clm in trainer if CLM is expected
    # Actually, for MMLU evaluation we want eval_dataset not eval_dataset_clm
    
    if source != original:
        set_cell_source(cell, source)

# Save the fixed notebook
with open('corrected-moe-FIXED-V4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("\nVariable references fixed!")

