import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")

def basic_clean(t: str) -> str:
    if not isinstance(t, str): return ""
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("@", t)   # preserva o símbolo
    t = HASHTAG_RE.sub("#", t)   # preserva o símbolo
    t = t.replace("\n"," ").strip()
    return t
