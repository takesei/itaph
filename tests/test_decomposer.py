import numpy as np
import pandas as pd
import pytest

from itaph.decomposer import Decomposer
from itaph.inventory import Inventory
from itaph.typing import Key, KeyList


@pytest.mark.smoke
class TestDecomposer:
    @pytest.fixture
    def keys(self):
        return KeyList(Key('A', 'x'), Key('A', 'y'), Key('B', 'x'), Key('B', 'y'))

    @pytest.fixture
    def value(self):
        return [
            [10, 10, 10, 10],
            [10, 20, 30, 40],
        ]

    def test_init(self, keys):
        c_t = pd.DataFrame(
            np.ones((4, 4)), columns=keys.to_index(), index=keys.to_index()
        )
        c_p = pd.DataFrame(np.ones(4), index=keys.to_index())
        c_s = pd.DataFrame(np.ones(4), index=keys.to_index())

        Decomposer(keys, c_t, c_p, c_s)

    def test_init_raise(self, keys):
        c_t = pd.DataFrame(
            np.ones((4, 4)), columns=keys.to_index(), index=keys.to_index()
        )
        c_p = pd.DataFrame(np.ones(4), index=keys.to_index())
        c_s = pd.DataFrame(np.ones(4), index=keys.to_index())

        with pytest.raises(ValueError):
            _c_t = pd.DataFrame(
                np.ones((4, 3)), columns=keys.to_index(), index=keys.to_index()
            )
            Decomposer(keys, _c_t, c_p, c_s)

        with pytest.raises(ValueError):
            _c_t = pd.DataFrame(
                np.ones((3, 3)), columns=keys.to_index(), index=keys.to_index()
            )
            Decomposer(keys, _c_t, c_p, c_s)

        with pytest.raises(ValueError):
            _c_p = pd.DataFrame(np.ones(3), index=keys.to_index())
            Decomposer(keys, c_t, _c_p, c_s)

        with pytest.raises(ValueError):
            _c_s = pd.DataFrame(np.ones(3), index=keys.to_index())
            Decomposer(keys, c_t, c_p, _c_s)

        with pytest.raises(KeyError):
            _c_t = pd.DataFrame(np.ones((4, 4)))
            _c_p = pd.DataFrame(np.ones(4))
            _c_s = pd.DataFrame(np.ones(4))
            Decomposer(keys, _c_t, _c_p, _c_s)


class TestDecomposerSlow:
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

        ret = dc.decompose(inp, out)

        T = ret.T.value  # noqa: N806
        p = ret.p.value
        s = ret.s.value
        sl_p = ret.sl_p.value
        sl_n = ret.sl_n.value

        assert ((T @ inp.vector + (p - s) - (sl_p + sl_n)) == out.vector).all()
        assert (T @ inp.vector).sum() == inp.vector.sum()

    def test_decompose_raise(self, keys, value):
        c_t = pd.DataFrame(
            np.ones((4, 4)), columns=keys.to_index(), index=keys.to_index()
        )
        c_p = pd.DataFrame(np.ones(4), index=keys.to_index())
        c_s = pd.DataFrame(np.ones(4), index=keys.to_index())

        dc = Decomposer(keys, c_t, c_p, c_s)

        inp = Inventory(keys, value[0])
        out = Inventory(
            KeyList(Key('A', 'x'), Key('A', 'y'), Key('B', 'x')), [10, 20, 30]
        )

        with pytest.raises(ValueError):
            dc.decompose(inp, out)
