import os

def fix_mojibake(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    with open(filename, 'rb') as f:
        data = f.read()

    # The issue is likely that UTF-8 bytes were interpreted as Latin-1
    # and then re-saved as UTF-8.
    # We can try to fix this by:
    # 1. Decoding as UTF-8 to get the mojibake string.
    # 2. Encoding as Latin-1 to get the original raw bytes.
    # 3. Decoding those raw bytes as UTF-8.
    
    try:
        # Step 1: Decode current "broken" file as UTF-8
        text = data.decode('utf-8')
        
        # Step 2 & 3: Attempt to reverse the "interpret as latin-1" mistake
        # This only works if all characters were in the Latin-1 range.
        fixed_data = text.encode('latin-1')
        fixed_text = fixed_data.decode('utf-8')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(fixed_text)
        print(f"Successfully recovered {filename} via latin-1/utf-8 roundtrip.")
    except Exception as e:
        print(f"Standard recovery failed: {e}. Falling back to manual replacements.")
        
        # Fallback: Manual replacement of common patterns
        # We'll use hex escapes to avoid shell/editor issues.
        mapping = {
            '\xc3\xa2\xe2\x82\xac\xe2\x80\x94': '\xe2\x80\x94', # em dash
            '\xc3\xa2\xe2\x82\xac\xe2\x80\x93': '\xe2\x80\x93', # en dash
            '\xc3\xa2\xca\x92\xc3\xa2\xe2\x80\xa6': '\xe2\x9c\x85', # checkmark
            '\xc3\xa2\xe2\x80\x94\xe2\x82\xac': '\xe2\x97\x80', # prev triangle
        }
        # Actually, if the file is already "double encoded", 
        # it's better to just read it, replace the specific bad strings, and write it.
        
        # Let's try the simplest manual mapping first.
        try:
            content = data.decode('utf-8', errors='replace')
            # Nav arrows
            content = content.replace('Ã¢â€”â‚¬', '◀')
            content = content.replace('Ã¢â€“Â¶', '▶')
            # Checkmarks/Status
            content = content.replace('Ã¢Å“â€¦', '✅')
            content = content.replace('Ã¢Å“â€œ', '✓')
            content = content.replace('Ã¢Ëœâ€˜', '☑')
            content = content.replace('Ã¢ËœÂ ', '☒')
            # Emojis
            content = content.replace('Ã°Å¸Å½Â¬', '🎬')
            content = content.replace('Ã°Å¸â€™Â¬', '💬')
            content = content.replace('Ã°Å¸â€ Å½', '🔍')
            content = content.replace('Ã°Å¸Â¤â€“', '🤖')
            content = content.replace('Ã°Å¸â€“Â¼Ã¯Â¸Â ', '🖼️')
            content = content.replace('Ã°Å¸â€œÂº', '📺')
            content = content.replace('Ã°Å¸Å’Â ', '🌐')
            content = content.replace('Ã°Å¸â€œâ€ž', '📄')
            content = content.replace('Ã°Å¸â€˜Â ', '🎬')
            # Timestamps/Other
            content = content.replace('Ã¢Â Â±', '🕒')
            content = content.replace('Ã¢Â Â­', '⏭')
            content = content.replace('Ã¢Â Â³', '⏳')
            content = content.replace('Ã¢â„¢Â»Ã¯Â¸Â ', '🔄')
            content = content.replace('Ã¢Â¬Å“', '⬜')
            content = content.replace('Ã¢â€ Â©', '↺')
            content = content.replace('Ã‚Â·', '·')
            content = content.replace('Ã¢â‚¬â€ ', '—')
            content = content.replace('Ã¢â‚¬â€œ', '–')
            content = content.replace('Ã¢â‚¬Â¦', '…')
            content = content.replace('Ã¢š ï¸ ', '⚠️')
            # Box drawing and UI icons
            content = content.replace('Ã¢â€ â‚¬', '─')
            content = content.replace('Ã¢Å¡â„¢Ã¯Â¸Â ', '⚙️')
            content = content.replace('Ã¢â€”â€¹', '○')
            content = content.replace('Ã¢Å“â€ ', '✂️')
            content = content.replace('Ã°Å¸â€™Â¡', '💡')
            content = content.replace('Ã¢Å“Â¨', '✨')
            content = content.replace('Ã°Å¸â€˜â‚¬', '👀')
            content = content.replace('Ã¢Å¡Â¡', '⚡')
            content = content.replace('Ã°Å¸â€œÂ±', '📱')
            content = content.replace('Ã°Å¸â€™Â»', '💻')
            content = content.replace('Ã¢â€”Â ', '●')
            content = content.replace('Ã°Å¸Â¤â€”', '🤗')
            content = content.replace('Ã°Å¸â€ Â¥', '🔥')
            content = content.replace('Ã°Å¸â€œÂ£', '📢')
            content = content.replace('Ã°Å¸â€œâ€š', '📂')
            content = content.replace('Ã°Å¸â€œâ€¹', '📋')
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Manual replacement finish for {filename}.")
        except Exception as e2:
            print(f"Critical failure: {e2}")

if __name__ == "__main__":
    fix_mojibake('app.py')
