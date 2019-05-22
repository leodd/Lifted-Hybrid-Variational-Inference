# https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set

import collections

class OrderedSet(collections.OrderedDict, collections.MutableSet):

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                 self.add(e)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)
    union = property(lambda self: self.__or__)


# import collections
#
# class OrderedSet(collections.MutableSet):
#
#     def __init__(self, iterable=None):
#         self.end = end = []
#         end += [None, end, end]         # sentinel node for doubly linked list
#         self.map = {}                   # key --> [key, prev, next]
#         if iterable is not None:
#             self |= iterable
#
#     def __len__(self):
#         return len(self.map)
#
#     def __contains__(self, key):
#         return key in self.map
#
#     def add(self, key):
#         if key not in self.map:
#             end = self.end
#             curr = end[1]
#             curr[2] = end[1] = self.map[key] = [key, curr, end]
#
#     def discard(self, key):
#         if key in self.map:
#             key, prev, next = self.map.pop(key)
#             prev[2] = next
#             next[1] = prev
#
#     def __iter__(self):
#         end = self.end
#         curr = end[2]
#         while curr is not end:
#             yield curr[0]
#             curr = curr[2]
#
#     def __reversed__(self):
#         end = self.end
#         curr = end[1]
#         while curr is not end:
#             yield curr[0]
#             curr = curr[1]
#
#     def pop(self, last=True):
#         if not self:
#             raise KeyError('set is empty')
#         key = self.end[1][0] if last else self.end[2][0]
#         self.discard(key)
#         return key
#
#     def __repr__(self):
#         if not self:
#             return '%s()' % (self.__class__.__name__,)
#         return '%s(%r)' % (self.__class__.__name__, list(self))
#
#     def __eq__(self, other):
#         if isinstance(other, OrderedSet):
#             return len(self) == len(other) and list(self) == list(other)
#         return set(self) == set(other)
#
#
# if __name__ == '__main__':
#     s = OrderedSet('abracadaba')
#     t = OrderedSet('simsalabim')
#     print(s | t)
#     print(s & t)
#     print(s - t)
