import pytest

from itaph.inventory import Inventory
from itaph.typing import Key, KeyList


@pytest.mark.smoke
class TestInventory:
    @pytest.fixture
    def keys(self):
        return [
            KeyList(Key('A', 'x'), Key('A', 'y'), Key('B', 'x'), Key('B', 'y')),
            KeyList(Key('A', 'x'), Key('A', 'y'), Key('C', 'x'), Key('C', 'y')),
        ]

    @pytest.fixture
    def value(self):
        return [
            [10, 10, 10, 10],
            [10, 20, 30, 40],
        ]

    def test_init(self, keys, value):
        for k in keys:
            for v in value:
                Inventory(k, v)

    def test_abel(self, keys, value):
        args = [
            ((keys[0], value[1]), (keys[0], value[0])),
            ((keys[1], value[1]), (keys[1], value[1])),
        ]

        exps_add = [
            [[20, 30], [40, 50]],
            [[20, 40], [60, 80]],
        ]

        exps_sub = [
            [[0, 10], [20, 30]],
            [[0, 0], [0, 0]],
        ]

        for (a1, a2), e in zip(args, exps_add, strict=False):
            i1 = Inventory(*a1)
            i2 = Inventory(*a2)

            ret = i1 + i2

            assert (ret.matrix.values == e).all()
            assert (ret.vector.values == sum(e, [])).all()
            assert ret.keys == i1.keys
            assert ret.keys == i2.keys

        for (a1, a2), e in zip(args, exps_sub, strict=False):
            i1 = Inventory(*a1)
            i2 = Inventory(*a2)

            ret = i1 - i2

            assert (ret.matrix.values == e).all()
            assert (ret.vector.values == sum(e, [])).all()
            assert ret.keys == i1.keys
            assert ret.keys == i2.keys

    def test_abel_assertion(self, keys, value):
        args = [
            ((keys[0], value[1]), (keys[1], value[0])),
        ]

        for a1, a2 in args:
            i1 = Inventory(*a1)
            i2 = Inventory(*a2)

            with pytest.raises(AssertionError):
                i1 + i2

            with pytest.raises(AssertionError):
                i1 - i2
