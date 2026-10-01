"""Test-only: route requests.post/get to openrouter.ai through curl (Python requests hangs on
this dev box's VPN, curl works — see the segment-library / storyboard notes). Never prints keys."""
import json as _json
import os
import subprocess
import tempfile

import requests

_real_post, _real_get = requests.post, requests.get


class _Resp:
    def __init__(self, status, body):
        self.status_code, self.text, self.content, self.headers = status, body, body.encode(), {}

    def json(self):
        return _json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            e = requests.HTTPError(f"{self.status_code} error")
            e.response = self
            raise e


def _curl(method, url, headers=None, json=None, timeout=None):
    read = (timeout[1] if isinstance(timeout, tuple) else timeout) or 120
    hdr = tempfile.NamedTemporaryFile("w", suffix=".hdr", delete=False)
    for k, v in (headers or {}).items():
        hdr.write(f"{k}: {v}\n")
    hdr.close()
    cmd = ["curl", "-s", "-m", str(int(read) + 10), "-X", method, url, "-H", f"@{hdr.name}",
           "-w", "\n__STATUS__%{http_code}"]
    body = None
    if json is not None:
        body = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        _json.dump(json, body)
        body.close()
        cmd += ["--data-binary", f"@{body.name}"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=int(read) + 20)
        out = r.stdout.decode("utf-8", "replace")
        text, _, status = out.rpartition("\n__STATUS__")
        return _Resp(int(status or 0), text)
    finally:
        os.remove(hdr.name)
        if body:
            os.remove(body.name)


def install():
    def post(url, headers=None, json=None, timeout=None, **kw):
        if "openrouter.ai" in url:
            return _curl("POST", url, headers, json, timeout)
        return _real_post(url, headers=headers, json=json, timeout=timeout, **kw)

    def get(url, headers=None, timeout=None, **kw):
        if "openrouter.ai" in url:
            return _curl("GET", url, headers, None, timeout)
        return _real_get(url, headers=headers, timeout=timeout, **kw)

    requests.post, requests.get = post, get
