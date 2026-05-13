import math
from core.output import filter_overlays_for_shots

def test_splitting_logic():
    # 10 shots, 3 per part
    shots = [{"timestamp": i*10, "id": i} for i in range(10)]
    overlays = [
        {"start_sec": 5, "highlight_text": "Overlay 1"}, # In part 1
        {"start_sec": 35, "highlight_text": "Overlay 2"}, # In part 2
        {"start_sec": 75, "highlight_text": "Overlay 3"}, # In part 3
        {"start_sec": 95, "highlight_text": "Overlay 4"}  # In part 4 (last shot is at 90, so this might be filtered out depending on padding)
    ]
    
    num_parts = 3
    total_shots = len(shots)
    chunk_size = math.ceil(total_shots / num_parts)
    
    print(f"Total shots: {total_shots}, Chunk size: {chunk_size}")
    
    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        shot_chunk = shots[start_idx:end_idx]
        if not shot_chunk: break
        
        chunk_ov = filter_overlays_for_shots(shot_chunk, overlays)
        
        print(f"Part {i+1}: Shots {start_idx} to {min(end_idx, total_shots)-1}")
        print(f"  Time range: {shot_chunk[0]['timestamp']} to {shot_chunk[-1]['timestamp']}")
        print(f"  Overlays: {[ov['highlight_text'] for ov in chunk_ov]}")

if __name__ == "__main__":
    test_splitting_logic()
