"""Per-shot Google images (Serper), the extras spare-swap repair, and chunked
overlay extraction with failure reasons."""

import os

import core.overlays_remotion as ovr
import core.pipeline as pl
import core.related_images as ri
import core.shot_images as si


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _serper_items(prefix, n, w=1920, h=1080):
    return {"images": [{"imageUrl": f"https://img/{prefix}{i}.jpg", "title": f"{prefix}{i}",
                        "imageWidth": w, "imageHeight": h, "link": f"https://page/{prefix}{i}"}
                       for i in range(n)]}


# ── Serper backend ────────────────────────────────────────────────────────────

def test_serper_search_parses_and_filters_small(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    payload = _serper_items("big", 2)
    payload["images"].append({"imageUrl": "https://img/small.jpg", "imageWidth": 300,
                              "imageHeight": 200})
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, json=json, headers=headers)
        return _Resp(payload)

    monkeypatch.setattr(ri.requests, "post", fake_post)
    out = ri.google_image_search("toyota camry", num=5)
    assert [o["url"] for o in out] == ["https://img/big0.jpg", "https://img/big1.jpg"]
    assert out[0]["context"] == "https://page/big0"
    assert seen["url"] == ri._SERPER_ENDPOINT
    assert seen["headers"]["X-API-KEY"] == "k"
    assert seen["json"]["q"] == "toyota camry"


def test_serper_retries_without_size_filter_on_400(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(dict(json))
        return _Resp({}, 400) if "tbs" in json else _Resp(_serper_items("a", 1))

    monkeypatch.setattr(ri.requests, "post", fake_post)
    out = ri.google_image_search("x", num=3)
    assert len(out) == 1
    assert "tbs" in calls[0] and "tbs" not in calls[1]


def test_serper_counts_as_configured(monkeypatch):
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "k")
    assert ri.cse_configured()
    monkeypatch.setenv("SERPER_API_KEY", "")
    assert not ri.cse_configured()


# ── per-shot images ───────────────────────────────────────────────────────────

def test_fallback_query_strips_video_words():
    shot = {"search_queries": ["slow motion close up Toyota engine bay 4k footage"]}
    assert si._fallback_query(shot) == "Toyota engine bay"


def test_fetch_shot_images_three_per_shot_no_repeats(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.delenv("SHOT_IMAGES_PER_SHOT", raising=False)
    shots = [
        {"slot_id": 1, "text": "The Camry engine", "search_queries": ["engine"]},
        {"slot_id": 2, "text": "Same engine again", "search_queries": ["engine"]},
        {"slot_id": 3, "text": "extra", "is_extra": True, "search_queries": ["x"]},
    ]
    # Both shots map to the SAME image query → one search, split without repeats.
    monkeypatch.setattr(si, "build_shot_image_queries",
                        lambda targets, *a, **k: {s["slot_id"]: "toyota camry engine"
                                                  for s in targets})
    searches = []

    def fake_search(q, num=5, errors=None, **k):
        searches.append(q)
        return [{"url": f"https://img/{i}.jpg", "title": "t", "context": f"https://p/{i}",
                 "width": 1920, "height": 1080} for i in range(10)]

    monkeypatch.setattr(si, "google_image_search", fake_search)
    fails = {"https://img/1.jpg"}   # a hotlink-blocked host → spare is used

    def fake_download(url, out_dir, index, stem=None):
        if url in fails:
            return ""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{stem}.jpg")
        open(path, "wb").write(b"x")
        return path

    monkeypatch.setattr(pl, "_download_image", fake_download)
    n = si.fetch_shot_images(shots, "proj", api_key="g")

    assert searches == ["toyota camry engine"]
    assert n == 6
    a = [i["url"] for i in shots[0]["images"]]
    b = [i["url"] for i in shots[1]["images"]]
    assert len(a) == 3 and len(b) == 3
    assert not set(a) & set(b)
    assert "https://img/1.jpg" not in a
    assert "images" not in shots[2]                       # extras are skipped
    assert all(os.path.exists(i["local_path"]) for i in shots[0]["images"])
    assert os.path.basename(os.path.dirname(shots[0]["images"][0]["local_path"])) == "shot_01"
    sources = open(tmp_path / "downloads" / "proj" / "images" / "shots" / "sources.txt",
                   encoding="utf-8").read()
    assert "https://p/0" in sources


def test_fetch_shot_images_noop_without_key(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    assert si.fetch_shot_images([{"slot_id": 1, "text": "x"}], "p") == 0


def test_build_queries_uses_llm_then_fallback(monkeypatch):
    import core.keywords as kw
    monkeypatch.setattr(kw, "_call_llm_json",
                        lambda *a, **k: {"queries": [{"slot_id": 1, "query": "Tesla Model 3"}]})
    shots = [{"slot_id": 1, "text": "a"},
             {"slot_id": 2, "text": "b", "search_queries": ["aerial drone shot highway"]}]
    q = si.build_shot_image_queries(shots, api_key=None, video_topic="EVs")
    assert q == {1: "Tesla Model 3", 2: "highway"}


# ── extras spare swap ─────────────────────────────────────────────────────────

def test_swap_failed_extras_uses_unused_spare():
    shots = [
        {"slot_id": 10, "is_extra": True, "extra_keyword": "camry review",
         "selected_results": [],     # pick failed and was purged
         "extra_spares": [{"url": "u-dead"}, {"url": "u-taken"}, {"url": "u-good"}]},
        {"slot_id": 11, "is_extra": True, "selected_results": [{"url": "u-taken"}]},
        {"slot_id": 1, "selected_results": []},   # a normal shot is left alone
    ]
    n = pl.swap_failed_extras(shots, blacklist={"u-dead"})
    assert n == 1
    assert shots[0]["selected_results"][0]["url"] == "u-good"
    assert shots[0]["selected_results"][0]["matched_query"] == "camry review"
    assert shots[2]["selected_results"] == []


def test_download_and_repair_replaces_failed_extra(monkeypatch, tmp_path):
    shots = [{"slot_id": 5, "is_extra": True, "priority": "low", "extra_keyword": "k",
              "selected_results": [{"url": "bad", "source": "youtube"}],
              "extra_spares": [{"url": "good", "source": "youtube"}]}]
    f = tmp_path / "clip.mp4"

    def fake_dl(shots, project_name, **kw):
        for s in shots:
            for c in s.get("selected_results") or []:
                if c["url"] == "good":
                    f.write_bytes(b"x")
                    c["local_path"] = str(f)
                    c["_dl_ok"] = True
                else:
                    c["_dl_failed"] = True
        return {"ok": 0, "failed": 0, "skipped": 0, "dir": str(tmp_path), "errors": []}

    monkeypatch.setattr(pl, "download_selected_clips", fake_dl)
    res = pl.download_and_repair(shots, "p", groq_key="g")
    assert res["repaired"] == 1 and res["ok"] == 1
    assert shots[0]["selected_results"][0]["url"] == "good"


# ── overlay chunking + failure reasons ───────────────────────────────────────

def _segs(total_sec, step=10):
    return [{"start": t, "end": t + step, "text": f"line {t}"} for t in range(0, total_sec, step)]


def test_chunk_segments_splits_long_transcript():
    chunks = ovr._chunk_segments(_segs(600), 150)
    assert len(chunks) == 4
    assert all(c[-1]["end"] - c[0]["start"] <= 150 for c in chunks)
    assert sum(len(c) for c in chunks) == 60


def test_extract_highlights_chunks_and_survives_one_failure(monkeypatch):
    import core.keywords as kw
    calls = []

    def fake_llm(client, system, user, **k):
        calls.append(user)
        if "part 2 of" in user:
            raise ValueError("truncated JSON")
        t = float(user.split("[")[1].split(" ")[0])
        return {"overlays": [{"text": f"STAT {int(t)}", "type": "stat",
                              "start": t, "end": t + 1}]}

    monkeypatch.setattr(kw, "_call_llm_json", fake_llm)
    diag = {}
    out = ovr.extract_overlay_highlights(_segs(600), groq_key=None, diag=diag)
    assert len(calls) == 4
    assert diag["chunks"] == 4 and diag["failed_chunks"] == 1
    assert "truncated" in diag["error"]
    assert len(out) == 3


def test_overlay_failure_reason_messages():
    assert "LLM call failed" in pl.overlay_failure_reason(
        {"chunks": 2, "failed_chunks": 2, "error": "Timeout", "highlights": 0, "rendered": 0})
    assert "Remotion" in pl.overlay_failure_reason(
        {"chunks": 1, "failed_chunks": 0, "highlights": 5, "rendered": 0, "remotion": False})
    assert "render" in pl.overlay_failure_reason(
        {"chunks": 1, "failed_chunks": 0, "highlights": 5, "rendered": 0, "remotion": True})
    assert pl.overlay_failure_reason({"rendered": 3}) == ""


# ── Backend failures: stop early, never leak keys ─────────────────────────────

def test_cse_429_trips_breaker_and_redacts_key(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "AIzaSECRET")
    monkeypatch.setenv("GOOGLE_CSE_CX", "cx1")
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["q"])
        return _Resp({}, 429)

    monkeypatch.setattr(ri.requests, "get", fake_get)
    errors = []
    for q in ("a", "b", "c"):
        assert ri.google_image_search(q, errors=errors) == []
    assert calls == ["a"]                       # later queries never hit the API
    assert len(errors) == 1 and "quota exhausted" in errors[0]
    assert ri.backend_tripped("cse")
    ri.reset_image_backends()
    assert not ri.backend_tripped("cse")


def test_redact_strips_query_string():
    msg = ("429 Client Error: Too Many Requests for url: "
           "https://www.googleapis.com/customsearch/v1?key=AIzaSECRET&cx=abc&q=x")
    out = ri._redact(msg)
    assert "AIzaSECRET" not in out and "cx=abc" not in out
    assert out.endswith("https://www.googleapis.com/customsearch/v1")


def test_serper_out_of_credits_trips_once(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    calls = []
    monkeypatch.setattr(ri.requests, "post",
                        lambda *a, **k: calls.append(1) or _Resp({}, 402))
    errors = []
    ri.google_image_search("a", errors=errors)
    ri.google_image_search("b", errors=errors)
    assert len(calls) == 1
    assert errors == ["google images: Serper account is out of credits — "
                      "stopped searching after 'a'"]


def test_shot_images_need_serper(monkeypatch, tmp_path):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_CX", "cx")
    monkeypatch.setattr(ri.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no CSE call")))
    errors = []
    shots = [{"slot_id": i, "text": "line", "search_queries": ["q"]} for i in range(1, 50)]
    assert si.fetch_shot_images(shots, "p", errors=errors) == 0
    assert len(errors) == 1 and "SERPER_API_KEY" in errors[0]


def test_bot_error_signature_hides_url_query():
    from bot.telegram_bot import _err_signature
    sig = _err_signature("google images 'x': 429 for url: "
                         "https://www.googleapis.com/customsearch/v1?key=AIzaSECRET&q=y")
    assert "AIzaSECRET" not in sig and "key=" not in sig


# ── Extras: say why there are none; repair failing extras from spares ─────────

def test_extras_extraction_failure_is_reported(monkeypatch):
    import core.extras as ex

    def boom(*a, **k):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(ex, "_call_llm_json", boom)
    errors, diag = [], {}
    assert ex.fetch_extra_shots("a script", "k", errors=errors, diag=diag) == []
    assert any("entity extraction failed" in e for e in errors)
    assert "found nothing" in ex.extras_empty_reason(diag)


def test_extras_entity_cache_shared_between_stages(monkeypatch):
    import core.extras as ex
    calls = []
    monkeypatch.setattr(ex, "_call_llm_json",
                        lambda *a, **k: calls.append(1) or {"brands": ["Ford"]})
    first = ex.extract_extra_entities("same script", "k")
    first["brands"].append("mutated")            # callers get a copy
    assert ex.extract_extra_entities("same script", "k")["brands"] == ["Ford"]
    assert len(calls) == 1


def test_extras_empty_reason_youtube():
    import core.extras as ex
    assert "returned nothing for all 4" in ex.extras_empty_reason(
        {"entities": 3, "keywords": 4, "no_results": 4, "all_filtered": 0, "clips": 0})
    assert ex.extras_empty_reason({"clips": 2}) == ""


def test_enforce_timeline_swaps_failing_extra_for_spare():
    bad = {"url": "https://yt/bad", "source": "youtube", "width": 1080, "height": 1920}
    good = {"url": "https://yt/good", "source": "youtube", "width": 1920, "height": 1080}
    shots = [{"slot_id": 9, "is_extra": True, "timestamp": 0, "extra_keyword": "ford logo",
              "selected_results": [bad], "extra_spares": [dict(bad), good]}]
    rep = pl.enforce_timeline(shots, errors=[])
    assert rep["ok"] and rep["dropped"] == 0
    assert shots[0]["selected_results"][0]["url"] == "https://yt/good"
