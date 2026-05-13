import re

def repair_utf8_in_latin1(text):
    """
    Attempts to fix text where UTF-8 bytes were interpreted as Latin-1.
    Example: 'Ã¢â‚¬â€ ' (E2 80 94) -> '—'
    """
    def fix_match(match):
        raw = match.group(0)
        try:
            # Re-encode as latin-1 to get original bytes, then decode as utf-8
            return raw.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return raw

    # This pattern matches sequences that look like multi-byte UTF-8 
    # encoded in Latin-1.
    # 2-byte: [C2-DF][80-BF]
    # 3-byte: [E0-EF][80-BF][80-BF]
    # 4-byte: [F0-F4][80-BF][80-BF][80-BF]
    pattern = r'[\u00c2-\u00df][\u0080-\u00bf]|[\u00e0-\u00ef][\u0080-\u00bf]{2}|[\u00f0-\u00f4][\u0080-\u00bf]{3}'
    
    return re.sub(pattern, fix_match, text)

def process_file(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Run the repair
    fixed_content = repair_utf8_in_latin1(content)
    
    # Also handle the specific weird ones that regex might miss
    manual = {
        'Ã¢â‚¬â€ ': '—',
        'Ã¢â‚¬â€œ': '–',
        'Ã‚Â·': '·'
    }
    for b, g in manual.items():
        fixed_content = fixed_content.replace(b, g)
        
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    print(f"Repaired {filename}")

if __name__ == "__main__":
    process_file('app.py')
