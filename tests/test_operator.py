import numpy as np
import pandas as pd
import pytest

from itaph.decomposer import Decomposer
from itaph.inventory import Inventory
from itaph.operator import Operator
from itaph.typing import Key, KeyList


@pytest.mark.smoke
class TestOperator:
    @pytest.fixture
    def keys(self):
        return [
            KeyList(Key('A', 'x'), Key('A', 'y'), Key('B', 'x'), Key('B', 'y')),
            KeyList(Key('A', 'x'), Key('A', 'y'), Key('C', 'x'), Key('C', 'y')),
            KeyList(
                Key('A', 'x'),
                Key('A', 'y'),
                Key('C', 'x'),
                Key('C', 'y'),
                Key('_affine', '_affine'),
            ),
        ]

    @pytest.fixture
    def value(self):
        return [
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            ),
            np.array(
                [
                    [1, 0, 0, 0, 5],
                    [0, 1, 0, 0, 5],
                    [0, 0, 1, 0, 5],
                    [0, 0, 0, 1, 5],
                    [0, 0, 0, 0, 1],
                ]
            ),
            np.array(
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 0],
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 0],
                    [0, 0, 0, 0, 1],
                ]
            ),
        ]

    def test_init(self, keys, value):
        for k in keys:
            for v in value:
                Operator(k, v)

    def test_init_raise(self, keys, value):
        with pytest.raises(ValueError):
            Operator(KeyList(Key('A', 'x'), Key('A', 'y')), value[0])

        with pytest.raises(ValueError):
            Operator(keys[0], value[0][:, :-1])

    def test_matmal(self, keys, value):
        key = keys[0]
        ops = [Operator(key, o) for o in value]
        inv = Inventory(key, [1, 2, 3, 4])
        exps = [[1, 2, 3, 4], [6, 7, 8, 9], [35, 80, 35, 80]]

        for op, exp in zip(ops, exps, strict=False):
            ret = op @ inv

            assert (ret.vector == exp).all()

    def test_mal(self, keys, value):
        key = keys[0]
        I, b, M = value  # noqa: E741
        o_i, o_b, o_m = [Operator(key, o) for o in value]

        assert ((o_i * o_i).params.values == I).all()
        assert ((o_i * o_b).params.values == b).all()
        assert ((o_i * o_m).params.values == M).all()

        assert (
            (o_b * o_b).params.values
            == [
                [1, 0, 0, 0, 10],
                [0, 1, 0, 0, 10],
                [0, 0, 1, 0, 10],
                [0, 0, 0, 1, 10],
                [0, 0, 0, 0, 1],
            ]
        ).all()
        assert (
            (o_b * o_m).params.values
            == [
                [1, 2, 3, 4, 10],
                [6, 7, 8, 9, 5],
                [1, 2, 3, 4, 10],
                [6, 7, 8, 9, 5],
                [0, 0, 0, 0, 1],
            ]
        ).all()

        assert (
            (o_m * o_b).params.values
            == [
                [1, 2, 3, 4, 55],
                [6, 7, 8, 9, 150],
                [1, 2, 3, 4, 55],
                [6, 7, 8, 9, 150],
                [0, 0, 0, 0, 1],
            ]
        ).all()


class TestOperatorSlow:
    @pytest.fixture
    def keys(self):
        return KeyList(Key('A', 'x'), Key('A', 'y'), Key('B', 'x'), Key('B', 'y'))

    @pytest.fixture
    def value(self):
        return [
            [10, 10, 10, 10],
            [10, 20, 30, 40],
        ]

    def test_decompose(self, keys, value):
        c_t = pd.DataFrame(
            np.ones((4, 4)), columns=keys.to_index(), index=keys.to_index()
        )
        c_p = pd.DataFrame(np.ones(4), index=keys.to_index())
        c_s = pd.DataFrame(np.ones(4), index=keys.to_index())

        dc = Decomposer(keys, c_t, c_p, c_s)

        inp = Inventory(keys, value[0])
        out = Inventory(keys, value[1])

        ret = Operator.decompose(dc, inp, out)

        assert ret.components is not None
