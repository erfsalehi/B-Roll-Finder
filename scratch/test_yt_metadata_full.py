import yt_dlp
import time

ydl_opts = {
    'extract_flat': False, # Get full info
    'quiet': True,
    'simulate': True,
    'skip_download': True
}

keyword = "tesla car driving"
start = time.time()
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    # Fetch ONLY 3 results to see how long it takes
    info = ydl.extract_info(f"ytsearch3:{keyword}", download=False)
    for i, entry in enumerate(info['entries']):
        print(f"Result {i+1}:")
        print(f"  Title: {entry.get('title')}")
        print(f"  Width: {entry.get('width')}")
        print(f"  Height: {entry.get('height')}")
        formats = entry.get('formats', [])
        resolutions = sorted(list(set(f.get('height') for f in formats if f.get('height'))), reverse=True)
        print(f"  Resolutions: {resolutions}")
        print("-" * 20)
print(f"Total time: {time.time() - start:.2f}s")
