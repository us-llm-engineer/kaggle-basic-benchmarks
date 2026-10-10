"""Label-free stylometric features of a sentence: length and word-shape statistics, punctuation rates, capitalisation, letter and
function-word frequencies. Nothing here uses labels, so the features can be computed for train and test at once."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

PUNCT = {"comma": ",", "semicolon": ";", "colon": ":", "period": ".", "exclaim": "!", "question": "?", "hyphen": "-", "dash": "—", "apostrophe": "'",
         "dquote": '"', "lparen": "(", "digit_marks": "0"}  # digit_marks counted separately below
ARCHAIC = ["thou", "thee", "thy", "thine", "hath", "doth", "whilst", "upon", "ere", "yet", "shall", "ye", "unto", "wherefore", "nay", "hither"]


def function_word_vocab(texts, min_count=300):
    """Stop words and archaic function words occurring at least `min_count` times in the corpus."""
    cv = CountVectorizer(vocabulary=sorted(set(ENGLISH_STOP_WORDS) | set(ARCHAIC)), token_pattern=r"(?u)\b\w+\b"); counts = np.asarray(cv.fit_transform(texts).sum(0)).ravel()
    return [w for w, n in zip(cv.get_feature_names_out(), counts) if n >= min_count]


def style_features(texts, fw_vocab):
    """DataFrame of stylometric features, one row per sentence."""
    s = pd.Series(list(texts)); n_chars = s.str.len().astype(float); words = s.str.findall(r"[A-Za-z']+"); n_words = words.str.len().clip(lower=1).astype(float)
    wl = words.apply(lambda ws: np.array([len(w) for w in ws]) if len(ws) else np.array([0]))
    F = pd.DataFrame({"chars": n_chars, "words": n_words, "mean_word_len": wl.apply(np.mean), "std_word_len": wl.apply(np.std), "max_word_len": wl.apply(np.max),
                      "chars_per_word": n_chars / n_words, "type_token": words.apply(lambda ws: len({w.lower() for w in ws}) / max(len(ws), 1))})
    for name, ch in PUNCT.items():
        if name != "digit_marks": F[f"{name}_per_100c"] = s.str.count(re.escape(ch)) / n_chars * 100
    F["digits_per_100c"] = s.str.count(r"\d") / n_chars * 100; F["commas_per_word"] = s.str.count(",") / n_words; F["semicolons"] = s.str.count(";")
    F["upper_frac"] = s.str.count(r"[A-Z]") / n_chars; F["cap_words_frac"] = words.apply(lambda ws: sum(w[0].isupper() for w in ws[1:]) / max(len(ws) - 1, 1)); F["allcaps_words"] = words.apply(lambda ws: sum(len(w) > 1 and w.isupper() for w in ws))
    F["starts_upper"] = s.str[0].str.isupper().astype(int); F["ends_punct"] = s.str[-1].isin(list(".!?")).astype(int); F["long_words_frac"] = wl.apply(lambda a: float((a >= 9).mean()))
    for i in range(1, 13): F[f"wl_{i}"] = wl.apply(lambda a, i=i: float((a == i).mean() if i < 12 else (a >= 12).mean()))
    low = s.str.lower()
    for ch in "abcdefghijklmnopqrstuvwxyz": F[f"letter_{ch}"] = low.str.count(ch) / n_chars
    fw = CountVectorizer(vocabulary=fw_vocab, token_pattern=r"(?u)\b\w+\b").fit_transform(low).toarray() / n_words.values[:, None]
    return pd.concat([F, pd.DataFrame(fw, columns=[f"fw_{w}" for w in fw_vocab])], axis=1)
