"""Machine learning: learn what each flatmate likes to eat.

For every person the app trains a small logistic-regression model (scikit-learn) on
  - positives: dishes they rated 👍 and their favourite recipes,
  - negatives: dishes they rated 👎,
  - background: a fixed random sample of other recipes, weighted lightly, so the model learns what
    sets a person's liked dishes apart from a typical recipe (this also lets it learn from 👍 alone).

Recipes are turned into features with a HashingVectorizer: the words of the title and ingredients plus
tokens for cuisine, course, diet and cooking-time bucket. The model then predicts, for every recipe,
the probability that the person will like it. How much that prediction counts grows with the number of
ratings (`confidence`), so the app starts with the fixed rules and relies more on learning over time.
Hard constraints (diet, allergies, dislikes) stay rules; the model only re-ranks dishes that are allowed.

No Streamlit and no database here, so this module can be tested on its own.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, HashingVectorizer
from sklearn.linear_model import LogisticRegression

import logic as L

N_FEATURES = 2 ** 15
BACKGROUND = 200          # random recipes used as "typical" examples
FULL_CONFIDENCE_AT = 8    # ratings + favourites needed before the model counts fully
MAX_BONUS = 5.0           # points a fully confident model can add or remove per person

COOKING_WORDS = {
    "cup", "cups", "tablespoon", "tablespoons", "teaspoon", "teaspoons", "tbsp", "tsp", "pound", "pounds", "ounce",
    "ounces", "chopped", "sliced", "diced", "minced", "fresh", "freshly", "ground", "large", "medium", "small",
    "taste", "divided", "needed", "optional", "finely", "coarsely", "thinly", "cut", "inch", "pieces", "piece",
    "salt", "pepper", "black", "water", "oil", "whole", "recipes", "recipe", "dishes", "dish", "main", "peeled",
    "drained", "rinsed", "packed", "softened", "melted", "room", "temperature", "beaten", "halved", "cubed",
    "trimmed", "plus", "more", "serving", "garnish", "each", "package", "can", "cans", "jar", "clove", "cloves",
    "allrecipes", "allstar", "allstars", "easy", "best", "quick", "homemade", "simple", "healthy", "low",
    "calorie", "fat", "sodium", "cholesterol", "high", "fiber", "kosher", "christmas", "easter", "day",
    "mother", "father", "july", "labor", "new", "year", "halloween", "chef", "john", "style", "white", "cooking",
    "purpose", "grated", "shredded", "unsalted", "boneless", "skinless", "frozen", "dried", "uncooked", "cooked",
    "light", "brown", "red", "green", "yellow", "extra", "virgin", "once", "twice", "cook", "eat", "make", "ahead",
    "leftovers", "dinner", "lunch", "recipes", "family", "favorite", "favorites", "ideas",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | COOKING_WORDS)

_vectorizer = HashingVectorizer(n_features=N_FEATURES, alternate_sign=False, norm="l2", binary=True,
                                token_pattern=r"(?u)\b[a-z][a-z_]+\b", stop_words=STOP_WORDS)
_analyzer = _vectorizer.build_analyzer()


def time_bucket(minutes) -> str:
    if not minutes:
        return "time_unknown"
    return "time_quick" if minutes <= 30 else "time_medium" if minutes <= 60 else "time_long"


def document(recipe: dict) -> str:
    """The text the model sees for one recipe."""
    tokens = [L.recipe_text(recipe), f"course_{recipe.get('course') or 'main'}", f"diet_{recipe.get('diet') or 'meat'}",
              time_bucket(recipe.get("time"))]
    if recipe.get("cuisine"):
        tokens.append("cuisine_" + recipe["cuisine"].replace(" ", "_"))
    return " ".join(tokens)


def featurize(recipes: list[dict]) -> sparse.csr_matrix:
    return _vectorizer.transform([document(r) for r in recipes])


class Features:
    """Feature rows for a fixed list of recipes (the recipe database), built once and reused."""

    def __init__(self, recipes: list[dict]):
        self.ids = [r["id"] for r in recipes]
        self.row = {rid: i for i, rid in enumerate(self.ids)}
        self.X = featurize(recipes) if recipes else sparse.csr_matrix((0, N_FEATURES))

    def matrix(self, recipes: list[dict]) -> sparse.csr_matrix:
        """Rows for any list of recipes: cached rows where possible, the rest computed now."""
        if not recipes:
            return sparse.csr_matrix((0, N_FEATURES))
        known = [self.row.get(r["id"]) for r in recipes]
        if all(k is not None for k in known):
            return self.X[known]
        rows = []
        for r, k in zip(recipes, known):
            rows.append(self.X[k] if k is not None else featurize([r]))
        return sparse.vstack(rows).tocsr()


@dataclass
class TasteModel:
    member: str
    clf: LogisticRegression
    n_liked: int
    n_disliked: int
    n_favs: int
    confidence: float
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    reference: np.ndarray = field(default_factory=lambda: np.array([]))  # sorted predictions for typical recipes

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        """Probability (0–1) that this person likes each recipe."""
        if X.shape[0] == 0:
            return np.zeros(0)
        return self.clf.predict_proba(X)[:, 1]

    def percentile(self, p: float) -> float:
        """Share of typical recipes this person would like less than this one (0 = least, 1 = most)."""
        if not len(self.reference):
            return p
        return float(np.searchsorted(self.reference, p)) / len(self.reference)

    def bonus(self, p: float) -> float:
        """Points added to (or removed from) a recipe's score: + for dishes above the person's typical taste,
        - for dishes below it, scaled by how much the model has learned so far."""
        return self.confidence * (MAX_BONUS * (2 * self.percentile(p) - 1) + p)


def train(member_id: str, ratings: dict[str, int], favs: list[str], recipes_by_id: dict[str, dict],
          features: Features, background_pool: list[str]) -> TasteModel | None:
    """Train one person's model. `ratings` maps recipe id -> +1 (👍) or -1 (👎); a rating beats a favourite."""
    pos = [rid for rid, v in ratings.items() if v > 0 and rid in recipes_by_id]
    neg = [rid for rid, v in ratings.items() if v < 0 and rid in recipes_by_id]
    fav_only = [rid for rid in favs if rid in recipes_by_id and rid not in ratings]
    pos += fav_only
    if not pos:
        return None  # nothing liked yet: nothing to learn from
    labelled = set(pos) | set(neg)
    rng = random.Random(zlib.crc32(member_id.encode()))
    pool = [rid for rid in background_pool if rid not in labelled]
    bg = rng.sample(pool, min(BACKGROUND, len(pool)))
    if not bg and not neg:
        return None

    ids = pos + neg + bg
    X = features.matrix([recipes_by_id[i] for i in ids])
    y = np.array([1] * len(pos) + [0] * (len(neg) + len(bg)))
    # explicit answers count fully; together the background weighs as much as the liked dishes
    w = np.array([1.0] * len(pos) + [1.0] * len(neg) + [len(pos) / max(len(bg), 1)] * len(bg))
    clf = LogisticRegression(C=5.0, max_iter=2000)
    clf.fit(X, y, sample_weight=w)

    n_signal = len(ratings) + len(fav_only)
    model = TasteModel(member=member_id, clf=clf, n_liked=len(pos) - len(fav_only), n_disliked=len(neg),
                       n_favs=len(fav_only), confidence=min(1.0, n_signal / FULL_CONFIDENCE_AT))
    model.likes, model.dislikes = explain(clf, [recipes_by_id[i] for i in pos + neg])
    if bg:
        model.reference = np.sort(clf.predict_proba(X[len(pos) + len(neg):])[:, 1])
    return model


def _pretty(token: str) -> str:
    diets = {"diet_meat": "meat dishes", "diet_pesce": "fish dishes", "diet_veg": "vegetarian dishes",
             "diet_vegan": "vegan dishes"}
    if token in diets:
        return diets[token]
    for prefix, fmt in (("cuisine_", "{} food"), ("course_", "{}")):
        if token.startswith(prefix):
            return fmt.format(token[len(prefix):].replace("_", " "))
    return {"time_quick": "quick dishes", "time_medium": "30–60 min dishes", "time_long": "long cooking",
            "time_unknown": ""}.get(token, token)


def explain(clf: LogisticRegression, recipes: list[dict], k: int = 6) -> tuple[list[str], list[str]]:
    """The words the model weighs most for and against, taken from the rated dishes themselves."""
    tokens = sorted({t for r in recipes for t in _analyzer(document(r))})
    if not tokens:
        return [], []
    idx = _vectorizer.transform(tokens).indices  # one hashed column per token (same order)
    coef = clf.coef_[0][idx]
    order = np.argsort(coef)
    likes = [_pretty(tokens[i]) for i in order[::-1] if coef[i] > 0.05][:k]
    dislikes = [_pretty(tokens[i]) for i in order if coef[i] < -0.05][:k]
    return [x for x in likes if x], [x for x in dislikes if x]
