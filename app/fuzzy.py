"""BK-tree for nearest-neighbour vocabulary lookup by Levenshtein distance.

Used at query time to correct typos: if a query token isn't in the indexed
vocabulary, we search for the closest match(es) within a small edit distance
and append them to the query token stream. The original token is kept too —
if the user happened to be right, BM25 will still find a match via exact
search.

Index-time only — the index already stores exact lemmas. Building the tree
is O(n log n) per insertion in practice; lookups are sub-linear in n.
"""
from __future__ import annotations


def _levenshtein(a: str, b: str) -> int:
    """Iterative DP Levenshtein distance. Both inputs assumed non-empty."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            if ca == cb:
                curr.append(prev[j])
            else:
                curr.append(1 + min(prev[j], prev[j + 1], curr[-1]))
        prev = curr
    return prev[-1]


class BKTree:
    """Burkhard-Keller tree keyed by Levenshtein distance.

    The tree is built incrementally — add() is safe to call repeatedly with
    duplicate words (they're ignored). search() returns words within
    `max_distance` of the query, including duplicates of `query` itself if
    they happen to be present.
    """

    __slots__ = ("_root", "_children")

    def __init__(self) -> None:
        self._root: str | None = None
        # word -> {edit_distance: child_word}
        self._children: dict[str, dict[int, str]] = {}

    def add(self, word: str) -> None:
        if not word:
            return
        if self._root is None:
            self._root = word
            self._children[word] = {}
            return
        node = self._root
        while True:
            d = _levenshtein(word, node)
            if d == 0:
                return  # already in tree
            children = self._children[node]
            if d in children:
                node = children[d]
            else:
                children[d] = word
                self._children[word] = {}
                return

    def search(self, query: str, max_distance: int) -> list[tuple[str, int]]:
        """Return [(word, distance)] for all indexed words within max_distance."""
        if self._root is None or not query:
            return []
        results: list[tuple[str, int]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = _levenshtein(query, node)
            if d <= max_distance:
                results.append((node, d))
            # Triangle inequality: only descend into children whose key is
            # within (d - max_distance, d + max_distance).
            lo, hi = d - max_distance, d + max_distance
            for cd, child in self._children[node].items():
                if lo <= cd <= hi:
                    stack.append(child)
        results.sort(key=lambda x: (x[1], x[0]))
        return results

    def __contains__(self, word: str) -> bool:
        if self._root is None or not word:
            return False
        node = self._root
        while True:
            d = _levenshtein(word, node)
            if d == 0:
                return True
            children = self._children[node]
            if d in children:
                node = children[d]
            else:
                return False

    def __len__(self) -> int:
        return len(self._children)


def build_bk_tree(token_stream: list[list[str]]) -> BKTree:
    """Build a BK-tree from the flat union of all indexed token lists.

    Words shorter than 3 chars are skipped — fuzzy matches on them produce
    too many false positives (CORS vs OOPS vs MOSS all at distance 2).
    """
    tree = BKTree()
    seen: set[str] = set()
    for tokens in token_stream:
        for tok in tokens:
            if len(tok) < 3 or tok in seen:
                continue
            seen.add(tok)
            tree.add(tok)
    return tree


def fuzzy_candidates(tree: BKTree, token: str, max_candidates: int = 3) -> list[str]:
    """Find likely typo corrections for `token`.

    Length-tuned edit budget:
      - length ≤ 8: max distance 1 (one edit before becoming ambiguous)
      - length 9+:  max distance 2 (long identifiers like
        OAuth2PasswordBearer tolerate more slop)

    Short words skipped entirely — too many false positives. The exact match
    is dropped from results so callers see only suggested corrections.
    """
    if not token or len(token) < 3:
        return []
    if len(token) <= 8:
        max_dist = 1
    else:
        max_dist = 2
    candidates = tree.search(token, max_dist)
    candidates = [(w, d) for w, d in candidates if d > 0]
    return [w for w, _ in candidates[:max_candidates]]
