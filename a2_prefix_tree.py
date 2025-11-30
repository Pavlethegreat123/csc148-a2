"""CSC148 Assignment 2: Autocompleter classes

=== CSC148 Fall 2025 ===
Department of Computer Science,
University of Toronto

=== Module Description ===
This file contains the definition of an Abstract Data Type (Autocompleter) and two
implementations of this interface, SimplePrefixTree and CompressedPrefixTree.
You'll complete both of these subclasses over the course of this assignment.

As usual, be sure not to change any parts of the given *public interface* in the
starter code---and this includes the instance attributes, which we will be
testing directly! You may, however, add new private attributes, methods, and
top-level functions to this file.
"""
from __future__ import annotations
from typing import Any
from python_ta.contracts import check_contracts

def find_prefix_tree(prefix: str, tree: SimplePrefixTree) -> SimplePrefixTree | None:
    ret_tree = None
    if not tree.root:
        for subtree in tree.subtrees:
            ret_tree = find_prefix_tree(prefix, subtree)
            if ret_tree is not None:
                return ret_tree
    if tree.is_leaf() or tree.is_empty():
        return None
    if tree.root[0] == prefix:
        return tree
    else:
        for subtree in tree.subtrees:
            ret_tree = find_prefix_tree(prefix, subtree)
    return ret_tree

def make_trees(prefix: list, weight: float) -> list[SimplePrefixTree]:
    """
    Returns a list of all the trees that can be made from the prefix
    >>> var = make_trees(['a', 'b', 'c'])
    """
    lst = []
    for letter in range(1, len(prefix) + 1):
        tree = SimplePrefixTree()
        tree.root = [" ".join(prefix[:letter])]; tree.weight = weight
        lst.append(tree)
    return lst

def dissolve_nested_tup_lst(obj: list|tuple):
    """
    >>> dissolve_nested_tup_lst([(1,5), [[(21,1)]], (3,1),[(4,2)]])
    [(1, 5), (21, 1), (3, 1), (4, 2)]
    """
    result = []
    for item in obj:
        if isinstance(item, tuple):
            result.append(item)
        elif isinstance(item, list):
            result.extend(dissolve_nested_tup_lst(item))

    return result

def decreasing_sort(tree: SimplePrefixTree) -> None:
    tree.subtrees.sort(key=lambda obj: obj.weight, reverse=True)
    for subtree in tree.subtrees:
        decreasing_sort(subtree)


################################################################################
# The Autocompleter ADT
################################################################################
class Autocompleter:
    """An abstract class representing the Autocompleter Abstract Data Type.
    """
    def __len__(self) -> int:
        """Return the number of values stored in this Autocompleter."""
        raise NotImplementedError

    def insert(self, value: Any, weight: float, prefix: list) -> None:
        """Insert the given value into this Autocompleter.

        The value is inserted with the given weight, and is associated with
        the prefix sequence <prefix>.

        If the value has already been inserted into this autocompleter
        (compare values using ==), then the given weight should be *added* to
        the existing weight of this value.

        Preconditions:
        - weight > 0
        - the given value is either:
            1) not in this Autocompleter, or
            2) was previously inserted with the SAME prefix sequence
        """
        raise NotImplementedError

    def autocomplete(self, prefix: list,
                     limit: int | None = None) -> list[tuple[Any, float]]:
        """Return up to <limit> matches for the given prefix.

        The return value is a list of tuples (value, weight), and must be
        sorted by non-increasing weight. You can decide how to break ties.

        If limit is None, return *every* match for the given prefix.

        Preconditions:
        - limit is None or limit > 0
        """
        raise NotImplementedError

    def remove(self, prefix: list) -> None:
        """Remove all values that match the given prefix.
        """
        raise NotImplementedError


################################################################################
# SimplePrefixTree (Tasks 1-3)
################################################################################
@check_contracts
class SimplePrefixTree(Autocompleter):
    """A simple prefix tree.

    Instance Attributes:
    - root:
        The root of this prefix tree.
        - If this tree is empty, <root> equals [].
        - If this tree is a leaf, <root> represents a value stored in the Autocompleter
          (e.g., 'cat').
        - If this tree is not a leaf and non-empty, <root> is a list representing a prefix
          (e.g., ['c', 'a']).
    - subtrees:
        A list of subtrees of this prefix tree.
    - weight:
        The weight of this prefix tree.
        - If this tree is empty, this equals 0.0.
        - If this tree is a leaf, this stores the weight of the value stored in the leaf.
        - If this tree is not a leaf and non-empty, this stores the *total weight* of
          the leaf weights in this tree.

    Representation invariants:
    - self.weight >= 0

    - (EMPTY TREE):
        If self.weight == 0.0, then self.root == [] and self.subtrees == [].
        This represents an empty prefix tree.
    - (LEAF):
        If self.subtrees == [] and self.weight > 0, then this tree is a leaf.
        (self.root is a value that was inserted into this tree.)
    - (NON-EMPTY, NON-LEAF):
        If self.subtrees != [], then self.root is a list (representing a prefix),
        and self.weight is equal to the sum of the weights of all leaves in self.

    - self.subtrees does not contain any empty prefix trees.
    - self.subtrees is *sorted* in non-increasing order of weight.
      (You can break ties any way you like.)
      Note that this applies to both leaves and non-leaf subtrees:
      both can appear in the same self.subtrees list, and both have a weight
      attribute.
    """
    root: Any
    weight: float
    subtrees: list[SimplePrefixTree]

    ###########################################################################
    # Part 1(a)
    ###########################################################################
    def __init__(self) -> None:
        """Initialize an empty simple prefix tree.
        """
        self.root = []
        self.subtrees = []
        self.weight = 0.0

    def is_empty(self) -> bool:
        """Return whether this simple prefix tree is empty."""
        return self.weight == 0.0

    def is_leaf(self) -> bool:
        """Return whether this simple prefix tree is a leaf."""
        if not self.subtrees and self.weight > 0:
            return True
        return False

    def __len__(self) -> int:
        """Return the number of LEAF values stored in this prefix tree.

        Note that this is a different definition than how we calculate __len__
        of regular trees from lecture!
        """
        num = 0
        if self.is_leaf():
            num += 1
        else:
            for subtree in self.subtrees:
                num += len(subtree)
        return num

    ###########################################################################
    # Extra helper methods
    ###########################################################################
    def __str__(self) -> str:
        """Return a string representation of this prefix tree.

        You may find this method helpful for debugging. You should not change this method
        (nor the helper _str_indented).
        """
        return self._str_indented()

    def _str_indented(self, depth: int = 0) -> str:
        """Return an indented string representation of this prefix tree.

        The indentation level is specified by the <depth> parameter.
        """
        if self.is_empty():
            return ''
        else:
            s = '  ' * depth + f'{self.root} ({self.weight})\n'
            for subtree in self.subtrees:
                s += subtree._str_indented(depth + 1)
            return s

    def _get_search_root(self, prefix: list) -> SimplePrefixTree:
        """Return the subtree from which to start autocomplete.

        If a matching prefix subtree exists (based on the first element of
        <prefix>), return that subtree; otherwise, return self.
        """
        if not prefix:
            return self
        tree = find_prefix_tree(prefix[0], self)
        if tree is not None:
            return tree
        return self

    def _autocomplete_no_limit(self, search_root: SimplePrefixTree,
                               prefix: list) -> list[list[tuple[Any, float]]]:
        """Helper for autocomplete when limit is None."""
        results = []
        for subtree in search_root.subtrees:
            results.append(subtree.autocomplete(prefix, None))
        return results

    def _autocomplete_with_limit(self, search_root: SimplePrefixTree,
                                 prefix: list, limit: int) -> list[list[tuple[Any, float]]]:
        """Helper for autocomplete when a limit is specified.

        This preserves the original behaviour of limiting by the number of
        *subtrees explored* at this level, not by total number of matches.
        """
        results = []
        count = 0
        for subtree in search_root.subtrees:
            if count >= limit:
                break
            results.append(subtree.autocomplete(prefix, limit))
            count += 1
        return results

    ###########################################################################
    # Add code for Parts 1(c), 2, and 3 here
    ###########################################################################

    def insert(self, value: Any, weight: float, prefix: list) -> None:
        """Insert the given value into this Tree.

        The value is inserted with the given weight, and is associated with
        the prefix sequence <prefix>.

        If the value has already been inserted into this autocompleter
        (compare values using ==), then the given weight should be *added* to
        the existing weight of this value.

        Preconditions:
        - weight > 0
        - the given value is either:
            1) not in this Autocompleter, or
            2) was previously inserted with the SAME prefix sequence
        """
        truth = False
        trees = make_trees(prefix, weight)
        if self.is_leaf():
            self.weight += weight
            return
        for subtree in trees:
            for tree in self.subtrees:
                if subtree.root == tree.root:
                    truth = True
                    tree.insert(value, weight, prefix)
                    break
        if not truth:
            val_tree = SimplePrefixTree(); val_tree.root = [value]; val_tree.weight = weight
            trees[-1].subtrees.append(val_tree)
            root_index = 0
            for subtree in trees:
                if subtree.root == self.root:
                    root_index = trees.index(subtree) + 1
                    break
            for x in range(root_index, len(trees) - 1):
                trees[x].subtrees.append(trees[x + 1])
            self.subtrees.append(trees[root_index])

        decreasing_sort(self)
        self.weight += weight

    def autocomplete(self, prefix: list,
                     limit: int | None = None) -> list[tuple[Any, float]]:
        """Return up to <limit> matches for the given prefix.

        The return value is a list of tuples (value, weight), and must be
        sorted by non-increasing weight. You can decide how to break ties.

        If limit is None, return *every* match for the given prefix.

        Preconditions:
        - limit is None or limit > 0
        >>> t = SimplePrefixTree()
        >>> t.insert('cat', 2.0, ['c', 'a', 't'])
        >>> t.insert('car', 3.0, ['c', 'a', 'r'])
        >>> t.insert('dog', 4.0, ['d', 'o', 'g'])
        >>> t.autocomplete(['d'], 1)
        [('dog', 4.0)]
        """
        # Base cases: leaf or empty tree
        if self.is_leaf():
            return [(self.root[0], self.weight)]
        if self.is_empty():
            return []

        # Decide which subtree to start searching from
        search_root = self._get_search_root(prefix)

        # Collect raw results, possibly with a limit
        if limit is None:
            raw_results = self._autocomplete_no_limit(search_root, prefix)
        else:
            raw_results = self._autocomplete_with_limit(search_root, prefix, limit)

        # Flatten nested lists of (value, weight) and sort by weight
        results = dissolve_nested_tup_lst(raw_results)
        results.sort(key=lambda t: t[1], reverse=True)
        return results

    def remove(self, prefix: list) -> None:
        """Remove all values that match the given prefix.
        """




################################################################################
# CompressedPrefixTree (Part 6)
################################################################################
@check_contracts
class CompressedPrefixTree(SimplePrefixTree):
    """A compressed prefix tree implementation.

    While this class has the same public interface as SimplePrefixTree,
    (including the initializer!) this version follows the definitions
    described in Part 6 of the assignment handout, which reduces the number of
    tree objects used to store values in the tree.

    Representation Invariants:
    - (NEW) This tree does not contain any compressible internal values.
    """
    subtrees: list[CompressedPrefixTree]  # Note the different type annotation

    ###########################################################################
    # Add code for Part 6 here
    ###########################################################################


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # Uncomment the python_ta lines below and run this module.
    # This is different that just running doctests! To run this file in PyCharm,
    # right-click in the file and select "Run a2_prefix_tree" or
    # "Run File in Python Console".
    #
    # python_ta will check your work and open up your web browser to display
    # its report. For full marks, you must fix all issues reported, so that
    # you see "None!" under both "Code Errors" and "Style and Convention Errors".
    # TIP: To quickly uncomment lines in PyCharm, select the lines below and press
    # "Ctrl + /" or "⌘ + /".
    # import python_ta
    # python_ta.check_all(config={
    #     'max-line-length': 100,
    #     'max-nested-blocks': 4
    # })
