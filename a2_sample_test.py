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
from a2_prefix_tree import SimplePrefixTree, CompressedPrefixTree
from a2_autocomplete_engines import SentenceAutocompleteEngine


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

    # Note that the returned tuples *must* be sorted in non-increasing weight
    # order. You can (and should) sort the tuples yourself inside
    # SimplePrefixTree.autocomplete.
    assert t.autocomplete([]) == [('dog', 4.0), ('car', 3.0), ('cat', 2.0)]

    # But keep in mind that the greedy algorithm here does not necessarily
    # return the highest-weight values!! In this case, the ['c'] subtree
    # is recursed on first.
    assert t.autocomplete([], 1) == [('car', 3.0)]


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

    # The trickiest part is that only *values* should be stored at leaves,
    # so even if you remove a specific prefix, its parent might get removed
    # from the tree as well!
    t.remove(['c', 'a'])

    assert len(t) == 1
    assert t.weight == 4.0

    # There is no more ['c'] subtree!
    assert len(t.subtrees) == 1
    assert t.subtrees[0].root == ['d']

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


###########################################################################
# Part 4 sample test (add your own for Parts 4 and 5!)
###########################################################################
def test_sentence_autocompleter() -> None:
    """Basic test for SentenceAutocompleteEngine.

    This test relies on the sample_sentences.csv dataset. That file consists
    of just a few lines, but there are three important details to notice:

        1. You should use the second entry of each csv file as the weight of
           the sentence. This entry can be a float! (Don't assume it's an int.)
        2. The file contains two sentences that are sanitized to the same
           string, and so this value is inserted twice. This means its weight
           is the *sum* of the weights from each of the two lines in the file.
        3. Numbers *are allowed* in the strings (this is true for both types
           of text-based autocomplete engines). Don't remove them!
    """
    engine = SentenceAutocompleteEngine({
        'file': 'data/texts/sample_sentences.csv',
        'autocompleter': 'simple'
    })

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


if __name__ == '__main__':
    import pytest
    pytest.main(['a2_sample_test.py'])
