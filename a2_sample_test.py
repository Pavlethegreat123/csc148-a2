"""CSC148 Assignment 2: Sample tests

=== CSC148 Fall 2025 ===
Department of Computer Science,
University of Toronto

=== Module description ===
This module contains sample tests for Assignment 2.

Warning: This is an extremely incomplete set of tests!
Add your own to practice writing tests and to be confident your code is correct.

Note: this file is for support purposes only, and is not part of your
assignment submission.
"""
from typing import Any

from a2_melody import Melody
from a2_prefix_tree import SimplePrefixTree, CompressedPrefixTree
from a2_autocomplete_engines import SentenceAutocompleteEngine, LetterAutocompleteEngine, MelodyAutocompleteEngine
import pytest

###########################################################################
# Parts 1(c) - 3 sample tests
###########################################################################
def test_simple_prefix_tree_structure() -> None:
    """This is a test for the structure of a small simple prefix tree.

    NOTE: This test should pass even if you insert these values in a different
    order. This is a good thing to try out.
    """
    t = SimplePrefixTree()
    t.insert('cat', 2.0, ['c', 'a', 't'])
    t.insert('car', 3.0, ['c', 'a', 'r'])
    t.insert('dog', 4.0, ['d', 'o', 'g'])
    print(t)

    # t has 3 values (note that __len__ only counts the inserted values,
    # which are stored at the *leaves* of the tree).
    assert len(t) == 3

    # t has a total weight of 9.0
    assert t.weight == 2.0 + 3.0 + 4.0

    # t has two subtrees, and order matters (because of weights).
    assert len(t.subtrees) == 2
    left = t.subtrees[0]
    right = t.subtrees[1]

    assert left.root == ['c']
    assert left.weight == 5.0

    assert right.root == ['d']
    assert right.weight == 4.0


def test_simple_prefix_tree_autocomplete() -> None:
    """This is a test for the correct autocomplete behaviour for a small
    simple prefix tree.

    NOTE: This test should pass even if you insert these values in a different
    order. This is a good thing to try out.
    """
    t = SimplePrefixTree()
    t.insert('cat', 2.0, ['c', 'a', 't'])
    t.insert('car', 3.0, ['c', 'a', 'r'])
    t.insert('dog', 4.0, ['d', 'o', 'g'])
    print(t)

    # Note that the returned tuples *must* be sorted in non-increasing weight
    # order. You can (and should) sort the tuples yourself inside
    # SimplePrefixTree.autocomplete.
    assert t.autocomplete([]) == [('dog', 4.0), ('car', 3.0), ('cat', 2.0)]

    # But keep in mind that the greedy algorithm here does not necessarily
    # return the highest-weight values!! In this case, the ['c'] subtree
    # is recursed on first.
    assert t.autocomplete([], 1) == [('car', 3.0)]

    assert t.autocomplete(['d'], 1) == [('dog', 4.0)]

def test_simple_prefix_tree_remove() -> None:
    """This is a test for the correct remove behaviour for a small
    simple prefix tree.

    NOTE: This test should pass even if you insert these values in a different
    order. This is a good thing to try out.
    """
    t = SimplePrefixTree()
    t.insert('cat', 2.0, ['c', 'a', 't'])
    t.insert('car', 3.0, ['c', 'a', 'r'])
    t.insert('dog', 4.0, ['d', 'o', 'g'])
    print(t)

    # The trickiest part is that only *values* should be stored at leaves,
    # so even if you remove a specific prefix, its parent might get removed
    # from the tree as well!
    t.remove(['c', 'a'])
    print(t)

    assert len(t) == 1
    assert t.weight == 4.0

    # There is no more ['c'] subtree!
    assert len(t.subtrees) == 1
    assert t.subtrees[0].root == ['d']

# chatGPT tests for insert

def _count_nodes(tree: SimplePrefixTree) -> int:
    """Return the number of non-empty nodes in the prefix tree.

    A "node" is a SimplePrefixTree object with weight > 0
    (i.e., not the initial empty tree before any insertions).
    """
    if tree.is_empty():
        return 0
    return 1 + sum(_count_nodes(sub) for sub in tree.subtrees)


def test_insert_empty_prefix_single_value_structure_and_len() -> None:
    """Inserting one value with an empty prefix [] into a new tree.

    Expected:
      - Tree has two nodes: internal root with prefix [], and a leaf node.
      - __len__ returns 1.
    """
    tree = SimplePrefixTree()
    tree.insert("cat", 1.0, [])

    # len counts only leaves
    assert len(tree) == 1

    # Root should be an internal node with prefix []
    assert tree.root == []
    assert not tree.is_leaf()
    assert not tree.is_empty()

    # Exactly one subtree, which should be the leaf
    assert len(tree.subtrees) == 1
    leaf = tree.subtrees[0]

    assert leaf.is_leaf()
    # Leaf root should be the stored value
    assert leaf.root == "cat"
    assert leaf.weight == pytest.approx(1.0)

    # Weight should be propagated to the root
    assert tree.weight == pytest.approx(1.0)

    # Two non-empty nodes: internal root and the leaf
    assert _count_nodes(tree) == 2


def test_insert_length_one_prefix_structure_and_len() -> None:
    """Inserting one value with a length-one prefix [x] into a new tree.

    Example: value 'cat' with prefix ['c'].

    Expected:
      - Tree has three nodes: internal [] -> internal ['c'] -> leaf 'cat'.
      - __len__ returns 1.
    """
    tree = SimplePrefixTree()
    tree.insert("cat", 2.5, ["c"])

    assert len(tree) == 1

    # Root internal node
    assert tree.root == []
    assert not tree.is_leaf()
    assert not tree.is_empty()
    assert len(tree.subtrees) == 1

    # Internal node for prefix ['c']
    prefix_node = tree.subtrees[0]
    assert not prefix_node.is_leaf()
    assert prefix_node.root == ["c"]
    assert len(prefix_node.subtrees) == 1

    # Leaf containing the value
    leaf = prefix_node.subtrees[0]
    assert leaf.is_leaf()
    assert leaf.root == "cat"
    assert leaf.weight == pytest.approx(2.5)

    # Weight propagation
    assert prefix_node.weight == pytest.approx(2.5)
    assert tree.weight == pytest.approx(2.5)

    # Three nodes: [], ['c'], and leaf 'cat'
    assert _count_nodes(tree) == 3


def test_insert_length_n_prefix_structure_and_len() -> None:
    """Inserting one value with a length-n prefix [x1, ..., xn].

    Example: prefix ['c', 'a', 't'] and value 'cat'.

    Expected:
      - Tree has (n + 2) nodes:
          internal [] ->
          internal ['c'] ->
          internal ['c', 'a'] ->
          internal ['c', 'a', 't'] ->
          leaf 'cat'
      - __len__ returns 1.
      - All internal nodes (and the leaf) have the correct weight.
    """
    tree = SimplePrefixTree()
    prefix = ["c", "a", "t"]
    value = "cat"
    w = 4.0

    tree.insert(value, w, prefix)

    # Only one value total
    assert len(tree) == 1

    # Root
    assert tree.root == []
    assert not tree.is_empty()
    assert not tree.is_leaf()

    # Node count check: n internal prefix nodes + root + leaf = n + 2
    assert _count_nodes(tree) == len(prefix) + 2

    # Walk down the chain of internal prefix nodes
    current = tree
    built_prefix: list[Any] = []
    for char in prefix:
        # exactly one subtree at each step, forming a chain
        assert len(current.subtrees) == 1
        current = current.subtrees[0]
        built_prefix.append(char)

        # Each of these should be non-leaf internal nodes with the accumulated prefix
        assert current.root == built_prefix
        assert not current.is_leaf()
        assert current.weight == pytest.approx(w)

    # After the loop, current is the last internal node with full prefix
    assert len(current.subtrees) == 1
    leaf = current.subtrees[0]
    assert leaf.is_leaf()
    assert leaf.root == value
    assert leaf.weight == pytest.approx(w)

    # Root weight also equals the leaf weight
    assert tree.weight == pytest.approx(w)


def test_weights_update_with_multiple_insertions_same_prefix() -> None:
    """Check that weights are updated correctly with multiple values.

    Insert two different values that share the same prefix ['c']:

      - 'cat' with weight 2.0
      - 'car' with weight 1.0

    Expected:
      - __len__ == 2 (two leaves).
      - Total weight at root and prefix node == 3.0.
      - Leaf weights remain 2.0 and 1.0.
    """
    tree = SimplePrefixTree()
    tree.insert("cat", 2.0, ["c"])
    tree.insert("car", 1.0, ["c"])

    assert len(tree) == 2
    assert tree.weight == pytest.approx(3.0)

    # There should be a single internal node for prefix ['c']
    assert len(tree.subtrees) == 1
    prefix_node = tree.subtrees[0]
    assert prefix_node.root == ["c"]
    assert prefix_node.weight == pytest.approx(3.0)

    # Two leaves under ['c']
    assert len(prefix_node.subtrees) == 2
    leaves = prefix_node.subtrees

    # Collect weights and roots for easier checking
    weights = sorted([leaf.weight for leaf in leaves], reverse=True)
    roots = {leaf.root for leaf in leaves}

    assert weights == [2.0, 1.0]
    assert roots == {"cat", "car"}


def test_weights_update_when_reinserting_same_value() -> None:
    """Reinserting the same value with the same prefix adds to its weight.

    Insert:
      - 'cat', weight 2.0, prefix ['c']
      - 'cat', weight 3.0, prefix ['c']

    Expected:
      - Only one leaf.
      - Leaf weight == 5.0.
      - Total tree and prefix-node weight == 5.0.
      - __len__ == 1.
    """
    tree = SimplePrefixTree()
    tree.insert("cat", 2.0, ["c"])
    tree.insert("cat", 3.0, ["c"])

    assert len(tree) == 1
    assert tree.weight == pytest.approx(5.0)

    assert len(tree.subtrees) == 1
    prefix_node = tree.subtrees[0]
    assert prefix_node.weight == pytest.approx(5.0)

    assert len(prefix_node.subtrees) == 1
    leaf = prefix_node.subtrees[0]
    assert leaf.is_leaf()
    assert leaf.root == "cat"
    assert leaf.weight == pytest.approx(5.0)

# chatgpt tests for autocomplete

def _build_basic_tree() -> SimplePrefixTree:
    """Return a small SimplePrefixTree for autocomplete testing."""
    t = SimplePrefixTree()
    t.insert('cat', 2.0, ['c', 'a', 't'])
    t.insert('car', 3.0, ['c', 'a', 'r'])
    t.insert('cap', 1.0, ['c', 'a', 'p'])
    t.insert('dog', 4.0, ['d', 'o', 'g'])
    t.insert('door', 5.0, ['d', 'o', 'o', 'r'])
    t.insert('doll', 1.5, ['d', 'o', 'l', 'l'])
    return t


###############################################################################
# 1. Number of matches for a prefix: 0, 1, many
###############################################################################

def test_autocomplete_no_matches_returns_empty_list() -> None:
    """Autocomplete on a prefix with 0 matches returns an empty list."""
    t = _build_basic_tree()
    result = t.autocomplete(['z'])

    assert result == []


def test_autocomplete_single_match_returns_that_match() -> None:
    """Autocomplete on a prefix that uniquely identifies one value.

    Here, only 'cat' should match the full prefix ['c', 'a', 't'].
    """
    t = _build_basic_tree()
    result = t.autocomplete(['c', 'a', 't'])

    assert len(result) == 1
    assert result[0][0] == 'cat'
    assert result[0][1] == pytest.approx(2.0)


def test_autocomplete_many_matches_no_limit_sorted_by_weight() -> None:
    """Autocomplete on a prefix with many matches and no limit.

    For prefix ['c'], we have 'car' (3.0), 'cat' (2.0), and 'cap' (1.0).
    They should be sorted by non-increasing weight.
    """
    t = _build_basic_tree()
    result = t.autocomplete(['c'])

    words = [pair[0] for pair in result]
    weights = [pair[1] for pair in result]

    assert set(words) == {'car', 'cat', 'cap'}
    assert weights == sorted(weights, reverse=True)


###############################################################################
# 2. Relationship between number of matches and the limit argument
###############################################################################

def test_autocomplete_limit_greater_than_number_of_matches() -> None:
    """If limit > number of matches, all matches are returned.

    For prefix ['c'], there are 3 matches, so limit=10 still returns 3.
    """
    t = _build_basic_tree()
    result = t.autocomplete(['c'], limit=10)

    assert len(result) == 3
    assert set(word for word, _ in result) == {'car', 'cat', 'cap'}


def test_autocomplete_limit_equal_to_number_of_matches() -> None:
    """If limit == number of matches, all matches are returned."""
    t = _build_basic_tree()
    # There are 3 matches for ['c'], so limit=3 should still return all 3.
    result = t.autocomplete(['c'], limit=3)

    assert len(result) == 3
    assert set(word for word, _ in result) == {'car', 'cat', 'cap'}


def test_autocomplete_limit_less_than_number_of_matches() -> None:
    """If limit < number of matches, *at most* limit results are returned.

    For prefix ['c'], there are 3 matches, so limit=2 returns 2 results.
    They must be the top 2 by weight: 'car' (3.0) and 'cat' (2.0).
    """
    t = _build_basic_tree()
    result = t.autocomplete(['c'], limit=2)

    assert len(result) == 2
    top_words = [word for word, _ in result]
    assert 'car' in top_words
    assert 'cat' in top_words
    # Ensure 'cap' (lowest weight) is not included
    assert 'cap' not in top_words


###############################################################################
# 3. When there are more matches than limit, check correct matches by weight
###############################################################################

def test_autocomplete_more_matches_than_limit_chooses_highest_weights() -> None:
    """When there are more matches than limit, return the highest-weight ones.

    We construct a tree where four words share the same prefix, with distinct
    weights, and use a small limit to make sure we get the correct top-k.
    """
    t = SimplePrefixTree()
    t.insert('alpha', 1.0, ['a'])
    t.insert('beta', 10.0, ['b'])
    t.insert('ape', 3.0, ['a'])
    t.insert('ant', 5.0, ['a'])
    t.insert('arch', 7.0, ['a'])

    # There are 4 matches for prefix ['a']: alpha, ape, ant, arch.
    # Their weights: 1.0, 3.0, 5.0, 7.0.
    # With limit=2, we expect the two highest-weight: 'arch' (7.0), 'ant' (5.0).
    result = t.autocomplete(['a'], limit=2)

    assert len(result) == 2
    words = {word for word, _ in result}
    assert words == {'arch', 'ant'}


def test_autocomplete_weight_updates_affect_ranking() -> None:
    """If a value is inserted multiple times, its cumulative weight is used.

    We make one of the lower-weight entries become the highest-weight by
    re-inserting it, and check that autocomplete reflects this.
    """
    t = SimplePrefixTree()
    t.insert('alpha', 1.0, ['a'])
    t.insert('ape', 3.0, ['a'])
    t.insert('ant', 5.0, ['a'])

    # Re-insert 'ape' with extra weight so its total becomes highest:
    # old 3.0 + new 5.0 = 8.0
    t.insert('ape', 5.0, ['a'])

    # Now weights: alpha=1.0, ant=5.0, ape=8.0.
    result = t.autocomplete(['a'], limit=2)

    assert len(result) == 2
    # First result should be 'ape' with highest weight
    assert result[0][0] == 'ape'
    assert result[0][1] == pytest.approx(8.0)
    # Second should be 'ant'
    assert result[1][0] == 'ant'
    assert result[1][1] == pytest.approx(5.0)


###########################################################################
# Part 4 sample test (add your own for Parts 4 and 5!)
###########################################################################
def test_letter_autocompleter() -> None:
    engine = LetterAutocompleteEngine({
        'file': 'data/texts/sample_words.txt',
        'autocompleter': 'simple'
    })
    print(engine.autocompleter)

    results = engine.autocomplete(['d', 'o'], 1)
    assert len(results) == 1
    assert results[0][0] == 'door'
    assert results[0][1] == 2.0

def test_sentence_autocompleter() -> None:
    engine = SentenceAutocompleteEngine({
        'file': 'data/texts/sample_sentences.csv',
        'autocompleter': 'complex'
    })
    print(engine.autocompleter)

    # Check simple autocompletion and sanitization
    results = engine.autocomplete('what a')
    assert len(results) == 1
    assert results[0][0] == 'what a wonderful world'
    assert results[0][1] == 1.0

    # Check that numbers are allowed in the sentences
    results = engine.autocomplete('numbers')
    assert len(results) == 1
    assert results[0][0] == 'numbers are 0k4y'

    # Check that one sentence can be inserted twice
    results = engine.autocomplete('a')
    assert len(results) == 1
    assert results[0][0] == 'a star is born'
    assert results[0][1] == 15.0 + 6.5

# def test_melody_autocompleter() -> None:
#     engine = MelodyAutocompleteEngine({
#         'file': 'data/melodies/more_melodies.csv',
#         'autocompleter': 'simple'
#     })
#     print(engine.autocompleter)
def test_melody_autocompleter_basic() -> None:
    engine = MelodyAutocompleteEngine({
        'file': 'data/melodies/songbook.csv',
        'autocompleter': 'simple'
    })

###########################################################################
# Part 6 sample tests
###########################################################################
def test_compressed_prefix_tree_structure() -> None:
    """This is a test for the correct structure of a compressed prefix tree.

    NOTE: This test should pass even if you insert these values in a different
    order. This is a good thing to try out.
    """
    t = CompressedPrefixTree()
    t.insert('cat', 2.0, ['c', 'a', 't'])
    t.insert('car', 3.0, ['c', 'a', 'r'])
    t.insert('dog', 4.0, ['d', 'o', 'g'])
    print("\n", t)

    # t has 3 values (note that __len__ only counts the values, which are
    # stored at the *leaves* of the tree).
    assert len(t) == 3

    # t has a total weight of 9.0
    assert t.weight == 2.0 + 3.0 + 4.0

    # t has two subtrees, and order matters (because of weights).
    assert len(t.subtrees) == 2
    left = t.subtrees[0]
    right = t.subtrees[1]

    # But note that the prefix values are different than for a SimplePrefixTree!
    assert left.root == ['c', 'a']
    assert left.weight == 5.0

    assert right.root == ['d', 'o', 'g']
    assert right.weight == 4.0

def _count_nodes(t: SimplePrefixTree) -> int:
    """Return the total number of non-empty nodes in the tree."""
    if t.is_empty():
        return 0
    if t.is_leaf():
        return 1
    return 1 + sum(_count_nodes(st) for st in t.subtrees)


def _collect_leaf_pairs(t: SimplePrefixTree) -> list[tuple[Any, float]]:
    """Return all (value, weight) tuples from leaves in t (order not guaranteed)."""
    if t.is_empty():
        return []
    if t.is_leaf():
        return [(t.root, t.weight)]
    result: list[tuple[Any, float]] = []
    for st in t.subtrees:
        result.extend(_collect_leaf_pairs(st))
    return result


###############################################################################
# Tests for sanitize and prefix helpers
###############################################################################


def test_sanitize_examples_from_docstring() -> None:
    # From your sanitize doctest
    assert sanitize("cAt") == "cat"
    assert sanitize("cA t") == "ca t"
    assert sanitize("cAt  ") == "cat"
    assert sanitize("cA-t") == "cat"
    assert sanitize("c8At") == "c8at"
    assert sanitize("n \\n muchluv") == "n  n muchluv"
    assert sanitize(" ") == ""
    assert sanitize("") == ""


def test_make_prefixes_letter() -> None:
    assert make_prefixes_letter("cat") == ["c", "a", "t"]
    # Spaces preserved
    assert make_prefixes_letter("a b") == ["a", " ", "b"]


def test_make_prefixes_word() -> None:
    assert make_prefixes_word("hello world") == ["hello", "world"]
    # Multiple spaces split into empty pieces that your engine won’t use
    assert make_prefixes_word("a  b") == ["a", "", "b"]


###############################################################################
# SimplePrefixTree: basic structure & insertion
###############################################################################


def test_simple_insert_empty_prefix_two_nodes() -> None:
    """Inserting with empty prefix [] creates root [] and one leaf."""
    t = SimplePrefixTree()
    t.insert("cat", 1.0, [])

    # Tree should have internal root [] and one leaf 'cat'
    assert t.root == []
    assert not t.is_leaf()
    assert len(t.subtrees) == 1
    leaf = t.subtrees[0]

    assert leaf.is_leaf()
    assert leaf.root == "cat"
    assert leaf.weight == pytest.approx(1.0)

    # Weight propagated to root
    assert t.weight == pytest.approx(1.0)

    # Two non-empty nodes: internal root + leaf
    assert _count_nodes(t) == 2
    assert len(t) == 1


def test_simple_insert_length_one_prefix_three_nodes() -> None:
    """Empty tree + one value with prefix [x] -> three nodes: [], [x], leaf."""
    t = SimplePrefixTree()
    t.insert("cat", 1.0, ["c"])

    assert t.root == []
    assert not t.is_leaf()
    assert len(t.subtrees) == 1

    mid = t.subtrees[0]
    assert not mid.is_leaf()
    assert mid.root == ["c"]
    assert len(mid.subtrees) == 1

    leaf = mid.subtrees[0]
    assert leaf.is_leaf()
    assert leaf.root == "cat"
    assert leaf.weight == pytest.approx(1.0)

    # Weights
    assert mid.weight == pytest.approx(1.0)
    assert t.weight == pytest.approx(1.0)

    # Two internal nodes + one leaf
    assert _count_nodes(t) == 3
    assert len(t) == 1


def test_simple_insert_length_n_prefix_n_plus_2_nodes() -> None:
    """Prefix [x1, ..., xn] should create (n+2) nodes including leaf."""
    t = SimplePrefixTree()
    prefix = ["c", "a", "t"]  # n = 3
    t.insert("cat", 2.0, prefix)

    # Path: [], ['c'], ['c','a'], ['c','a','t'], leaf
    assert t.root == []
    assert t.subtrees[0].root == ["c"]
    assert t.subtrees[0].subtrees[0].root == ["c", "a"]
    assert t.subtrees[0].subtrees[0].subtrees[0].root == ["c", "a", "t"]

    leaf = t.subtrees[0].subtrees[0].subtrees[0].subtrees[0]
    assert leaf.is_leaf()
    assert leaf.root == "cat"

    assert _count_nodes(t) == 5  # n + 2 with n=3
    assert len(t) == 1
    assert t.weight == pytest.approx(2.0)


def test_simple_insert_multiple_updates_weights_and_len() -> None:
    """Check weights and len after multiple insertions and duplicates."""
    t = SimplePrefixTree()
    t.insert("cat", 2.0, ["c", "a", "t"])
    t.insert("car", 3.0, ["c", "a", "r"])
    t.insert("dog", 4.0, ["d", "o", "g"])
    # Duplicate value with same prefix should increase weight but not len
    t.insert("cat", 1.0, ["c", "a", "t"])

    leaf_pairs = sorted(_collect_leaf_pairs(t), key=lambda p: p[0])
    assert leaf_pairs == [
        ("car", pytest.approx(3.0)),
        ("cat", pytest.approx(3.0)),  # 2.0 + 1.0
        ("dog", pytest.approx(4.0)),
    ]

    assert len(t) == 3
    assert t.weight == pytest.approx(3.0 + 3.0 + 4.0)


###############################################################################
# SimplePrefixTree: autocomplete (unlimited and limited, by spec)
###############################################################################


def _build_basic_simple_tree() -> SimplePrefixTree:
    """Helper: build tree for 'car', 'cat', 'care', 'dog', 'danger'."""
    t = SimplePrefixTree()
    t.insert("car", 1.0, ["c", "a", "r"])
    t.insert("cat", 2.0, ["c", "a", "t"])
    t.insert("care", 3.0, ["c", "a", "r", "e"])
    t.insert("dog", 4.0, ["d", "o", "g"])
    t.insert("danger", 5.0, ["d", "a", "n", "g", "e", "r"])
    return t


def test_simple_autocomplete_no_matches() -> None:
    t = _build_basic_simple_tree()
    assert t.autocomplete(["x"]) == []


def test_simple_autocomplete_prefix_ca_unlimited() -> None:
    t = _build_basic_simple_tree()
    results = t.autocomplete(["c", "a"])
    # Expect 'car', 'care', 'cat' in non-increasing weight order
    assert [v for (v, _) in results] == ["danger"][:0]  # just to enforce list type
    names = [v for (v, _) in results]
    weights = [w for (_, w) in results]

    assert set(names) == {"car", "cat", "care"}
    # Must be sorted by non-increasing weight
    assert weights == sorted(weights, reverse=True)


def test_simple_autocomplete_empty_prefix_returns_all() -> None:
    t = _build_basic_simple_tree()
    results = t.autocomplete([])
    assert set(v for (v, _) in results) == {"car", "cat", "care", "dog", "danger"}


def test_simple_autocomplete_limit_basic() -> None:
    """Limit smaller than number of matches (simple case)."""
    t = SimplePrefixTree()
    t.insert("a", 5.0, ["a"])
    t.insert("b", 4.0, ["b"])
    t.insert("c", 3.0, ["c"])

    results = t.autocomplete([], limit=2)
    # Top 2 weights: 'a' (5.0) and 'b' (4.0)
    assert len(results) == 2
    assert [v for (v, _) in results] == ["a", "b"]


###############################################################################
# SimplePrefixTree: remove
###############################################################################


def test_simple_remove_specific_leaf() -> None:
    t = SimplePrefixTree()
    t.insert("cat", 1.0, ["c", "a", "t"])
    t.insert("car", 1.0, ["c", "a", "r"])
    t.insert("dog", 2.0, ["d", "o", "g"])

    t.remove(["c", "a", "t"])  # Remove only 'cat'

    assert len(t) == 2
    names = {v for (v, _) in _collect_leaf_pairs(t)}
    assert names == {"car", "dog"}
    assert t.weight == pytest.approx(1.0 + 2.0)


def test_simple_remove_internal_prefix_makes_tree_empty() -> None:
    t = SimplePrefixTree()
    t.insert("cat", 1.0, ["c", "a", "t"])
    t.insert("car", 1.0, ["c", "a", "r"])

    t.remove(["c", "a"])

    assert t.is_empty()
    assert t.subtrees == []
    assert len(t) == 0
    assert t.weight == pytest.approx(0.0)


def test_simple_remove_internal_prefix_partial() -> None:
    """Remove ['c','a'] when other values ('dog') remain."""
    t = SimplePrefixTree()
    t.insert("cat", 1.0, ["c", "a", "t"])
    t.insert("car", 1.0, ["c", "a", "r"])
    t.insert("dog", 2.0, ["d", "o", "g"])

    t.remove(["c", "a"])

    assert not t.is_empty()
    assert len(t) == 1
    pairs = _collect_leaf_pairs(t)
    assert pairs == [("dog", pytest.approx(2.0))]
    assert t.root == []
    assert len(t.subtrees) == 1  # Only dog subtree remains


def test_simple_remove_empty_prefix_clears_tree() -> None:
    t = _build_basic_simple_tree()
    t.remove([])

    assert t.is_empty()
    assert t.subtrees == []
    assert len(t) == 0
    assert t.weight == pytest.approx(0.0)


def test_simple_remove_nonexistent_prefix_no_change() -> None:
    t = _build_basic_simple_tree()
    before_pairs = sorted(_collect_leaf_pairs(t))
    before_weight = t.weight

    t.remove(["x", "y", "z"])

    assert sorted(_collect_leaf_pairs(t)) == before_pairs
    assert t.weight == pytest.approx(before_weight)


###############################################################################
# CompressedPrefixTree: structure & behaviour
###############################################################################


def _build_basic_compressed_tree() -> CompressedPrefixTree:
    t = CompressedPrefixTree()
    t.insert("cat", 2.0, ["c", "a", "t"])
    t.insert("car", 3.0, ["c", "a", "r"])
    t.insert("dog", 4.0, ["d", "o", "g"])
    return t


def test_compressed_prefix_tree_structure_small() -> None:
    """Compressed tree for 'cat','car','dog' should match handout example."""
    t = _build_basic_compressed_tree()

    # Three values total
    assert len(t) == 3
    assert t.weight == pytest.approx(2.0 + 3.0 + 4.0)

    # Root has two subtrees: ['c', 'a'] and ['d','o','g'] (order by weight)
    assert len(t.subtrees) == 2
    left, right = t.subtrees

    # left and right order depends on weights (5 vs 4)
    if left.root == ["d", "o", "g"]:
        left, right = right, left

    assert left.root == ["c", "a"]
    assert left.weight == pytest.approx(5.0)

    assert right.root == ["d", "o", "g"]
    assert right.weight == pytest.approx(4.0)
    assert right.subtrees[0].is_leaf()
    assert right.subtrees[0].root == "dog"


def test_compressed_autocomplete_matches_simple() -> None:
    """Behaviour of autocomplete should match SimplePrefixTree with same values."""
    simple = _build_basic_simple_tree()
    compressed = CompressedPrefixTree()
    # Insert same values into compressed via public API
    compressed.insert("car", 1.0, ["c", "a", "r"])
    compressed.insert("cat", 2.0, ["c", "a", "t"])
    compressed.insert("care", 3.0, ["c", "a", "r", "e"])
    compressed.insert("dog", 4.0, ["d", "o", "g"])
    compressed.insert("danger", 5.0, ["d", "a", "n", "g", "e", "r"])

    for prefix in ([], ["c"], ["c", "a"], ["d"], ["d", "a"]):
        s_results = simple.autocomplete(prefix, limit=3)
        c_results = compressed.autocomplete(prefix, limit=3)
        assert s_results == c_results


def test_compressed_remove_matches_simple() -> None:
    """Remove on compressed should have same effect as on simple."""
    simple = _build_basic_simple_tree()
    compressed = CompressedPrefixTree()
    compressed.insert("car", 1.0, ["c", "a", "r"])
    compressed.insert("cat", 2.0, ["c", "a", "t"])
    compressed.insert("care", 3.0, ["c", "a", "r", "e"])
    compressed.insert("dog", 4.0, ["d", "o", "g"])
    compressed.insert("danger", 5.0, ["d", "a", "n", "g", "e", "r"])

    for prefix in (["c", "a"], ["d", "o", "g"], []):
        simple.remove(prefix)
        compressed.remove(prefix)
        assert sorted(_collect_leaf_pairs(simple)) == sorted(
            _collect_leaf_pairs(compressed._simple)  # type: ignore[attr-defined]
        )


###############################################################################
# LetterAutocompleteEngine
###############################################################################


def test_letter_engine_basic(tmp_path) -> None:
    # Prepare a small text file with messy capitalization/punctuation
    content = "Cat\ncar!!\n\n DOG  \n###\n"
    file_path = tmp_path / "words.txt"
    file_path.write_text(content, encoding="utf8")

    engine = LetterAutocompleteEngine(
        {"file": str(file_path), "autocompleter": "simple"}
    )

    # Sanitization + insertion
    results_c = engine.autocomplete("c")
    names_c = {s for (s, _) in results_c}
    assert names_c == {"cat", "car"}

    results_d = engine.autocomplete("d")
    assert results_d == [("dog", pytest.approx(1.0))]

    # Empty or whitespace prefix after sanitize -> no matches
    assert engine.autocomplete("", limit=5) == []


###############################################################################
# SentenceAutocompleteEngine
###############################################################################


def test_sentence_engine_basic(tmp_path) -> None:
    # CSV: "sentence,weight"
    content = "Hello world,2.5\nhello there!!!,1.0\nhow are you,3.0\n"
    file_path = tmp_path / "sentences.csv"
    file_path.write_text(content, encoding="utf8")

    engine = SentenceAutocompleteEngine(
        {"file": str(file_path), "autocompleter": "simple"}
    )

    # After sanitize/parse:
    #   "Hello world"   -> "hello world" weight 2.5
    #   "hello there!!!"-> "hello there" weight 1.0
    #   "how are you"   -> "how are you" weight 3.0

    res_hello = engine.autocomplete("hello")
    hellos = {s for (s, _) in res_hello}
    assert hellos == {"hello world", "hello there"}

    res_hello_w = engine.autocomplete("hello w")
    assert res_hello_w == [("hello world", pytest.approx(2.5))]

    res_how = engine.autocomplete("how are")
    assert res_how == [("how are you", pytest.approx(3.0))]


###############################################################################
# MelodyAutocompleteEngine
###############################################################################


def test_melody_engine_basic(tmp_path) -> None:
    # Each row: name, pitch1,dur1,pitch2,dur2,...
    # Melody1 notes: (60,500),(62,500),(64,500) -> intervals [2,2]
    # Melody2 notes: (60,500),(59,500),(60,500) -> intervals [-1,1]
    csv_content = "Song1,60,500,62,500,64,500\nSong2,60,500,59,500,60,500\n"
    file_path = tmp_path / "melodies.csv"
    file_path.write_text(csv_content, encoding="utf8")

    engine = MelodyAutocompleteEngine(
        {"file": str(file_path), "autocompleter": "simple"}
    )

    # Autocomplete on first interval 2 -> Song1 only
    res_2 = engine.autocomplete([2])
    assert len(res_2) == 1
    m1, w1 = res_2[0]
    assert isinstance(m1, Melody)
    assert m1.name == "Song1"
    assert w1 == pytest.approx(1.0)

    # Autocomplete on entire [2,2] sequence -> Song1
    res_22 = engine.autocomplete([2, 2])
    assert len(res_22) == 1
    assert res_22[0][0].name == "Song1"

    # prefix [-1] -> Song2
    res_m1 = engine.autocomplete([-1])
    assert len(res_m1) == 1
    assert res_m1[0][0].name == "Song2"


if __name__ == '__main__':
    import pytest
    pytest.main(['a2_sample_test.py'])
