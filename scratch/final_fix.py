import os

def final_fix(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Define the broken sequences exactly as they appear in the file
    # We use a list of tuples to handle potential overlapping
    replacements = [
        ('â€”â€”', '◀'),
        ('â€”', '—'),
        ('â€“', '–'),
        ('âšâ„¢ï¸', '⚙️'),
        ('âœ…', '✅'),
        ('â Â­', '⏭'),
        ('â Â³', '⏳'),
        ('âš ï¸', '⚠️'),
        ('âž¤', '▶'),
        ('â–¶', '▶'),
        ('Â · Â ', ' · '),
        ('Â·', '·'),
        ('? Suggest', '✨ Suggest'),
        ('? Preview', '🎬 Preview'),
        ('? Watch', '📺 Watch'),
        ('? Details', '📄 Details'),
        ('? Script:', '💬 Script:'),
        ('? Keywords:', '🔍 Keywords:'),
        ('? Searched:', '🔍 Searched:'),
        ('? Tweak', '🔄 Tweak'),
        ('? Finish', '✅ Finish'),
        ('ðŸ"Å’', '📌'),
        ('ðŸ"š', '📦'),
        ('ðŸ"', '📄'),
        ('ââ‚¬¢', '•')
    ]

    for bad, good in replacements:
        content = content.replace(bad, good)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Final fix applied to {filename}")

if __name__ == "__main__":
    final_fix('app.py')
