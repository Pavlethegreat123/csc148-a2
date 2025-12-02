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

from idlelib.autocomplete import AutoComplete
from typing import Any
from python_ta.contracts import check_contracts

def find_if_leaf_exist(value: Any, tree: Autocompleter) -> bool:
    truth = False
    for subtree in tree:
        if subtree == value:
            truth = True
        else:
            truth = find_prefix_tree(value, subtree)
    return truth


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
    >>> var = make_trees(['a', 'b', 'c'], 1.0)

    """
    lst = []
    for letter in range(1, len(prefix) + 1):
        tree = SimplePrefixTree()
        root = []
        for letter in prefix[:letter]:
            root.append(letter)
        # tree.root = [" ".join(prefix[:letter])]
        tree.root = root
        tree.weight = weight

        lst.append(tree)
    return lst


def dissolve_nested_tup_lst(obj: list | tuple):
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

    def _remove_helper(self, prefix):
        # Base cases: empty tree or leaf can't contain an internal prefix node.
        if self.is_empty() or self.is_leaf():
            return 0.0

        total_removed = 0.0
        new_subtrees = []

        for subtree in self.subtrees:
            # Case 1: this subtree is exactly the internal node representing <prefix>.
            # Remove the entire subtree and add its weight to total_removed.
            if (not subtree.is_leaf()) and subtree.root == prefix:
                total_removed += subtree.weight
                # We do NOT keep this subtree.

            else:
                # Case 2: recurse into the subtree to remove any matching descendants.
                removed_from_child = subtree._remove_helper(prefix)

                if removed_from_child > 0.0:
                    # Child's own weight has ALREADY been adjusted in its helper.
                    total_removed += removed_from_child

                # Only keep the child if it is still non-empty.
                if not subtree.is_empty():
                    new_subtrees.append(subtree)

        # Update this node's subtrees and weight.
        new_subtrees.sort(key=lambda st: st.weight, reverse=True)
        self.subtrees = new_subtrees
        self.weight -= total_removed

        return total_removed

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

    def _collect_all(self, limit: int | None) -> list[tuple[Any, float]]:
        """Collect all (value, weight) pairs in this subtree, greedily by weight.

        Respects <limit> if given.
        """
        if self.is_empty():
            return []
        if self.is_leaf():
            return [(self.root, self.weight)] if (limit is None or limit > 0) else []

        results: list[tuple[Any, float]] = []
        for st in self.subtrees:
            if limit is not None and len(results) >= limit:
                break
            sublimit = None if limit is None else (limit - len(results))
            results.extend(st._collect_all(sublimit))
        return results

    def _autocomplete_helper(
        self, prefix: list, limit: int | None
    ) -> list[tuple[Any, float]]:
        """Recursive helper implementing prefix matching and greedy limiting."""
        if self.is_empty():
            return []
        if self.is_leaf():
            # If we reached a leaf, the prefix has already matched on the way down.
            return [(self.root, self.weight)]

        # Case 1: self.root is a prefix of the desired prefix
        if len(prefix) >= len(self.root) and prefix[:len(self.root)] == self.root:
            if len(prefix) == len(self.root):
                # We've reached the internal node representing exactly <prefix>.
                return self._collect_all(limit)
            else:
                # Need to go deeper.
                next_prefix = prefix[:len(self.root) + 1]
                for st in self.subtrees:
                    if not st.is_leaf() and st.root == next_prefix:
                        return st._autocomplete_helper(prefix, limit)
                # No matching child
                return []

        # Case 2: desired prefix is shorter than self.root, but matches its start.
        # Then this whole subtree matches the prefix.
        if len(prefix) < len(self.root) and self.root[:len(prefix)] == prefix:
            return self._collect_all(limit)

        # No match.
        return []


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
        # Every node on the path gets its weight increased by <weight>.
        self.weight += weight

        # Case 1: this node's prefix equals the full prefix sequence.
        # We either update an existing leaf with this value, or create a new one.
        if self.root == prefix:
            # Try to find an existing leaf storing this value.
            for st in self.subtrees:
                if st.is_leaf() and st.root == value:
                    # Duplicate value with same prefix: bump its weight and re-sort.
                    st.weight += weight
                    self.subtrees.sort(key=lambda t: t.weight, reverse=True)
                    return

            # No existing leaf with this value: create a new leaf.
            leaf = SimplePrefixTree()
            leaf.root = value  # LEAF root is the *value*, not [value]
            leaf.weight = weight
            leaf.subtrees = []

            self.subtrees.append(leaf)
            self.subtrees.sort(key=lambda t: t.weight, reverse=True)
            return

        # Case 2: we are at an internal node whose prefix is a proper prefix of <prefix>.
        # The child we want has prefix one element longer than this node's prefix.
        next_len = len(self.root) + 1
        child_prefix = prefix[:next_len]

        # Try to find an existing child with this prefix.
        child = None
        for st in self.subtrees:
            # Internal nodes have list prefixes in .root; leaves have a non-list root.
            if not st.is_leaf() and st.root == child_prefix:
                child = st
                break

        # If no such child exists, create it.
        if child is None:
            child = SimplePrefixTree()
            child.root = child_prefix
            child.subtrees = []
            child.weight = 0.0  # recursive insert will add <weight>
            self.subtrees.append(child)

        # Recurse into the child.
        child.insert(value, weight, prefix)

        # Maintain invariant: subtrees sorted by non-increasing weight.
        self.subtrees.sort(key=lambda t: t.weight, reverse=True)

        #  my shit ##

        # truth = False
        # trees = make_trees(prefix, weight)
        # if self.is_leaf():
        #     self.weight += weight
        #     return
        # for subtree in trees:
        #     if truth:
        #         break
        #     for tree in self.subtrees:
        #         print("insert tree: ", subtree.root, "self tree: ", tree.root)
        #         if subtree.root == tree.root:
        #             truth = True
        #             tree.insert(value, weight, prefix)
        #             break
        #
        # if not truth:
        #     val_tree = SimplePrefixTree(); val_tree.root = [value]; val_tree.weight = weight
        #     trees[-1].subtrees.append(val_tree)
        #     root_index = 0
        #     for subtree in trees:
        #         if subtree.root == self.root:
        #             root_index = trees.index(subtree) + 1
        #             break
        #     for x in range(root_index, len(trees) - 1):
        #         trees[x].subtrees.append(trees[x + 1])
        #
        #
        #     self.subtrees.append(trees[root_index])
        #
        # decreasing_sort(self)
        # self.weight += weight
        #
        # print(self)


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
        raw_results = self._autocomplete_helper(prefix, limit)

        # Sort the matches we actually collected by non-increasing weight
        raw_results.sort(key=lambda t: t[1], reverse=True)

        # Enforce limit at the end as well, just in case
        if limit is not None and len(raw_results) > limit:
            raw_results = raw_results[:limit]

        return raw_results

    def remove(self, prefix: list) -> None:
        """Remove all values that match the given prefix.
        """
        # Removing with empty prefix means "remove everything".
        if prefix == []:
            self.root = []
            self.subtrees = []
            self.weight = 0.0
            return

        if self.is_empty():
            return

            # This will recursively remove matching subtrees and adjust weights.
        self._remove_helper(prefix)

        # If everything was removed, reset to the empty-tree representation.
        if self.weight == 0.0:
            self.root = []
            self.subtrees = []


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
