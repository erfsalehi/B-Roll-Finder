"""Related still images (the /images path): subject → image-query rules, the
Google Custom Search call (resolution filtering), the library-only run, and the
guarantee that stills never enter the video candidate pool."""

import pytest

import core.related_images as ri


@pytest.fixture(autouse=True)
def _no_serper(monkeypatch):
    # These tests exercise the Custom Search backend; a real SERPER_API_KEY in the
    # environment would route searches to Serper instead.
    monkeypatch.delenv("SERPER_API_KEY", raising=False)


# ── image-query rules ─────────────────────────────────────────────────────────

def test_build_image_queries_covers_all_categories():
    ents = {"brands": ["Toyota"], "models": ["Camry"], "parts": ["alternator"],
            "products": ["Sea Foam"], "themes": ["gas station forecourt"]}
    qs = ri.build_image_queries(ents)
    kinds = {q["kind"] for q in qs}
    assert kinds == {"brand", "model", "part", "product", "idea"}
    text = " | ".join(q["query"].lower() for q in qs)
    assert "toyota logo" in text
    assert "camry" in text
    assert "alternator" in text
    assert "sea foam" in text
    assert "gas station forecourt" in text


def test_build_image_queries_caps_and_round_robins():
    # 10 products x 2 templates = 20 queries; a cap of 12 must still give every
    # product its FIRST ('product') query before spending budget on seconds.
    products = [f"Item{i}" for i in range(10)]
    qs = ri.build_image_queries({"products": products}, max_queries=12)
    assert len(qs) == 12
    firsts = {q["query"].lower().replace(" product", "")
              for q in qs if q["query"].lower().endswith("product")}
    assert firsts == {p.lower() for p in products}   # all 10 covered


def test_build_image_queries_dedupes():
    qs = ri.build_image_queries({"brands": ["Toyota", "Toyota"]})
    assert len({q["query"].lower() for q in qs}) == len(qs)


# ── Google Custom Search call ─────────────────────────────────────────────────

class _FakeResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _cse_payload(items):
    return {"items": items}


def test_google_image_search_filters_small_images(monkeypatch):
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_CX", "cx")
    items = [
        {"link": "https://x/big.jpg", "title": "big",
         "image": {"width": 1920, "height": 1080, "thumbnailLink": "t1"}},
        {"link": "https://x/small.jpg", "title": "small",
         "image": {"width": 640, "height": 360, "thumbnailLink": "t2"}},
    ]
    monkeypatch.setattr(ri.requests, "get",
                        lambda url, params=None, timeout=None: _FakeResp(_cse_payload(items)))
    out = ri.google_image_search("toyota logo", num=5)
    urls = [o["url"] for o in out]
    assert "https://x/big.jpg" in urls
    assert "https://x/small.jpg" not in urls          # dropped: below 1280x720
    assert all(o["source"] == "google_image" for o in out)


def test_google_image_search_no_credentials_returns_empty(monkeypatch):
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)
    errs = []
    assert ri.google_image_search("anything", errors=errs) == []
    assert errs and "GOOGLE_CSE" in errs[0]


def test_fetch_related_images_dedupes_across_queries(monkeypatch):
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_CX", "cx")
    monkeypatch.setattr(ri, "extract_extra_entities",
                        lambda script, key, **k: {"brands": ["Toyota"], "models": [],
                                             "parts": [], "products": [], "themes": []})
    # Every query returns the SAME image URL — the final list must de-dup it.
    monkeypatch.setattr(ri, "google_image_search",
                        lambda term, num=4, errors=None: [
                            {"url": "https://x/same.jpg", "source": "google_image",
                             "title": term, "thumbnail": "", "width": 1920, "height": 1080}])
    imgs = ri.fetch_related_images("script about Toyota", api_key="k")
    assert len(imgs) == 1
    assert imgs[0]["url"] == "https://x/same.jpg"
    assert imgs[0]["kind"] == "brand"


# ── run_related_images: library-only, no timeline ─────────────────────────────

def test_run_related_images_downloads_to_library(monkeypatch, tmp_path):
    import core.pipeline as pipeline
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_CX", "cx")

    monkeypatch.setattr("core.transcription.transcribe_audio",
                        lambda path, key: [{"start": 0.0, "end": 1.0, "text": "Toyota Camry"}])
    monkeypatch.setattr("core.related_images.fetch_related_images",
                        lambda script, key, errors=None, per_query=None: [
                            {"url": "https://x/a.jpg", "query": "toyota logo", "kind": "brand",
                             "title": "toyota", "thumbnail": ""}])
    # Pretend the download succeeds by writing a real file.
    def _fake_dl(url, out_dir, index):
        import os
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, f"img-{index:03d}.jpg")
        with open(fp, "wb") as f:
            f.write(b"jpeg")
        return fp
    monkeypatch.setattr(pipeline, "_download_image", _fake_dl)

    stored = []
    monkeypatch.setattr("core.clip_library.store_clip",
                        lambda desc, data, **kw: stored.append((desc, data)) or True)

    res = pipeline.run_related_images("audio.mp3", project_name="imgtest")
    assert res["n_images"] == 1
    assert res["subjects"] == ["toyota logo"]
    assert "xml_path" not in res                       # library-only, no timeline
    # Stored with the image source so it stays out of the video candidate pool.
    assert stored and stored[0][1]["source"] == "google_image"


def test_run_related_images_requires_credentials(monkeypatch, tmp_path):
    import core.pipeline as pipeline
    import pytest
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)
    with pytest.raises(ValueError):
        pipeline.run_related_images("audio.mp3", project_name="x")


# ── stills never injected into the video candidate pool ───────────────────────

def test_inject_library_skips_images(monkeypatch):
    import core.clip_library as cl
    monkeypatch.setattr(cl, "search_library", lambda q, top_k=5: [
        {"url": "https://vid/clip.mp4", "original_source": "youtube"},
        {"url": "https://img/pic.jpg", "original_source": "google_image"},
    ])
    shots = [{"shot_intent": "engine bay", "priority": "normal", "video_results": []}]
    injected = cl.inject_library_candidates(shots, top_k=5)
    urls = [r.get("url") for r in shots[0]["video_results"]]
    assert "https://vid/clip.mp4" in urls
    assert "https://img/pic.jpg" not in urls           # image filtered out
    assert injected == 1
