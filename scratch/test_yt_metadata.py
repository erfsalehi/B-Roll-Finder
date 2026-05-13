import yt_dlp

ydl_opts = {
    'extract_flat': True,
    'quiet': True,
    'simulate': True
}

keyword = "tesla car driving"
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(f"ytsearch5:{keyword}", download=False)
    for i, entry in enumerate(info['entries']):
        print(f"Result {i+1}:")
        print(f"  Title: {entry.get('title')}")
        print(f"  Width: {entry.get('width')}")
        print(f"  Height: {entry.get('height')}")
        print(f"  Resolution: {entry.get('resolution')}")
        print(f"  Keys: {list(entry.keys())}")
        print("-" * 20)
