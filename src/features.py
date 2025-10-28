def build_numeric_features(s):
    """
    Gera métricas simples a partir de um texto.
    Lida automaticamente com valores NaN, None e tipos não string.
    """
    # garante que o valor é string
    if not isinstance(s, str):
        if s is None:
            s = ""
        else:
            s = str(s)
    s = s or ""

    return {
        "len": len(s),
        "n_hash": s.count("#"),
        "n_mention": s.count("@"),
        "n_exc": s.count("!"),
        "n_q": s.count("?"),
        "n_url_like": int(("http" in s) or ("www" in s)),
        "upper_ratio": (sum(c.isupper() for c in s) + 1) / (len(s) + 1)
    }
