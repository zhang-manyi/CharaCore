"""Small HTTPS Responses/Chat Completions adapter. No network until an explicit call."""
import json
import os
from pathlib import Path
import re
from urllib import error, parse, request

from characore.protocol import digest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "CHARACORE_JUDGE_"
FIELDS = {"BASE_URL", "API_KEY", "MODEL", "MAX_TOKENS", "TIMEOUT_SECONDS",
          "MAX_CALLS", "MAX_INPUT_BYTES", "JSON_MODE", "API_STYLE"}


def settings(env_file=None):
    """Read literal KEY=value settings; never source shell code or expand variables."""
    values = {}
    path = Path(env_file) if env_file else ROOT / ".env"
    if env_file and not path.is_file():
        raise ValueError("specified env file does not exist")
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                raise ValueError("env file requires literal KEY=value lines")
            if value.startswith(('"', "'")):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError("unclosed quote in env file")
                value = value[1:-1]
            if key.startswith(PREFIX) and key[len(PREFIX):] not in FIELDS:
                raise ValueError("unknown judge configuration field")
            if key in values:
                raise ValueError("duplicate env configuration field")
            values[key] = value
    for key in FIELDS:
        name = PREFIX + key
        if name in os.environ:
            values[name] = os.environ[name]
    return values


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward credentials to a different endpoint.


class APIResponseError(RuntimeError):
    def __init__(self, message, response_text=None):
        super().__init__(message)
        self.response_text = response_text


class APIJudge:
    def __init__(self, env_file=None, allow_calls=False):
        values = settings(env_file)
        def required(key):
            value = values.get(PREFIX + key, "")
            if not value or "\n" in value or "\r" in value:
                raise ValueError(f"missing or invalid {PREFIX + key}")
            return value
        self._key = required("API_KEY")
        url = required("BASE_URL").rstrip("/")
        parsed = parse.urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
                parsed.password or parsed.query or parsed.fragment):
            raise ValueError("judge base URL must be HTTPS without credentials/query/fragment")
        self.api_style = values.get(PREFIX + "API_STYLE", "responses")
        if self.api_style not in ("responses", "chat_completions"):
            raise ValueError("API_STYLE must be responses or chat_completions")
        self.endpoint = url + ("/responses" if self.api_style == "responses" else "/chat/completions")
        self.model = required("MODEL")
        self.allow_calls = allow_calls
        self.calls = 0
        def integer(key, default, maximum):
            try:
                value = int(values.get(PREFIX + key, str(default)))
            except ValueError:
                raise ValueError(f"{PREFIX + key} must be an integer") from None
            if not 1 <= value <= maximum:
                raise ValueError(f"{PREFIX + key} must be 1..{maximum}")
            return value
        self.max_tokens = integer("MAX_TOKENS", 4096, 16384)
        self.timeout = integer("TIMEOUT_SECONDS", 120, 600)
        self.max_calls = integer("MAX_CALLS", 100, 100000)
        self.input_limit = integer("MAX_INPUT_BYTES", 16384, 1000000)
        mode = values.get(PREFIX + "JSON_MODE", "true")
        if mode not in ("true", "false"):
            raise ValueError("JSON_MODE must be true or false")
        self.json_mode = mode == "true"
        self.metadata = dict(kind="remote_judge_api", api_style=self.api_style, endpoint=self.endpoint, model=self.model,
                             decoding="provider defaults; no temperature override", max_output_tokens=self.max_tokens, json_mode=self.json_mode,
                             adapter_sha256=digest(__file__))
        self.budget = dict(max_calls_per_process=self.max_calls, timeout_seconds=self.timeout,
                           max_input_bytes=self.input_limit)

    def redact(self, text):
        return text.replace(self._key, "[REDACTED]")

    def check_input(self, messages, input_limit=None):
        """Byte budget. The caller's token budget does not apply to a remote judge."""
        size = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        if size > self.input_limit:
            raise ValueError("API input byte budget exceeded; refusing truncation")

    def __call__(self, messages):
        if not self.allow_calls:
            raise ValueError("API disabled; explicitly pass --allow-api to send judge material")
        self.check_input(messages)
        if self.calls >= self.max_calls:
            raise ValueError("API request budget exhausted")
        if self.api_style == "responses":
            payload = dict(model=self.model, input=messages, max_output_tokens=self.max_tokens,
                           stream=False, store=False)
            if self.json_mode:
                payload["text"] = {"format": {"type": "json_object"}}
        else:
            payload = dict(model=self.model, messages=messages, max_completion_tokens=self.max_tokens,
                           stream=False, store=False)
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}
        req = request.Request(self.endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                              headers={"Content-Type": "application/json",
                                       "Authorization": "Bearer " + self._key,
                                       "User-Agent": "CharaCore-judge/0.1"},
                              method="POST")
        self.calls += 1
        body_text = None
        try:
            with request.build_opener(NoRedirect()).open(req, timeout=self.timeout) as response:
                body = response.read(4_000_001)
            if len(body) > 4_000_000:
                raise ValueError("API response exceeded size budget")
            body_text = self.redact(body.decode("utf-8"))
            data = json.loads(body_text)
            if self.api_style == "responses":
                raw = "".join(part["text"] for item in data.get("output", [])
                              if item.get("type") == "message" and item.get("role") == "assistant"
                              for part in item.get("content", []) if part.get("type") == "output_text")
                finish = data.get("status")
            else:
                raw = data["choices"][0]["message"]["content"]
                finish = data["choices"][0].get("finish_reason")
            if not isinstance(raw, str):
                raise ValueError("API returned no text content")
            return raw, dict(provider_response=data, provider_response_text=body_text,
                             usage=data.get("usage"), response_model=data.get("model"),
                             finish_reason=finish,
                             request_number=self.calls)
        except error.HTTPError as exc:
            # Keep a redacted body for diagnosing API protocol errors, never request headers.
            try:
                safe_body = self.redact(exc.read(65536).decode("utf-8", errors="replace"))
            except Exception:
                safe_body = None
            raise APIResponseError(f"judge API HTTP {exc.code}; check saved response", safe_body) from None
        except Exception as exc:
            raise APIResponseError(f"judge API failure ({type(exc).__name__}); no response accepted", body_text) from None
