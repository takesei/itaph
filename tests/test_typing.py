import pandas as pd
import pytest

from itaph.typing import Key, KeyList, ProblemVariables


@pytest.mark.smoke
class TestKeyList:
    def test_key_list_init(self):
        args = [
            [],
            [Key('a', 'A'), Key('b', 'B')],
            [Key('a', 'A'), Key('c', 'C')],
        ]

        for arg in args:
            KeyList(*arg)

    def test_key_list_eq(self):
        args = [
            (
                [Key('a', 'A'), Key('b', 'B')],
                [Key('a', 'A'), Key('b', 'B')],
            ),
            (
                [Key('a', 'A'), Key('b', 'B')],
                [Key('b', 'B'), Key('a', 'A')],
            ),
            (
                [Key('a', 'A'), Key('b', 'B')],
                [Key('a', 'A'), Key('c', 'C')],
            ),
            (
                [Key('a', 'A'), Key('b', 'B')],
                [Key('a', 'A')],
            ),
        ]

        exps = [True, False, False, False]

        for (a1, a2), e in zip(args, exps, strict=False):
            kl1 = KeyList(*a1)
            kl2 = KeyList(*a2)

            assert (kl1 == kl2) is e, f'{kl1} == {kl2}, expects {e}, got {kl1 == kl2}'

    def test_key_list_assert(self):
        args = [
            (
                [Key('a', 'A'), Key('b', 'B')],
                [Key('c', 'A'), Key('b', 'B')],
            ),
            (
                [Key('a', 'A'), Key('b', 'B')],
                [Key('a', 'A'), Key('c', 'C')],
            ),
        ]

        with pytest.raises(AssertionError):
            for a1, a2 in args:
                kl1 = KeyList(*a1)
                kl2 = KeyList(*a2)

                kl1.assert_eq(kl2)

    def test_key_list_add(self):
        args = [
            [Key('a', 'A')],
            [Key('a', 'A'), Key('c', 'C')],
        ]

        for arg in args:
            kl = KeyList()
            kl2 = kl.add(*arg)

            assert kl != kl2

        for arg in args:
            kl = KeyList(Key('b', 'B'))
            kl2 = kl.add(*arg)

            assert kl != kl2

    def test_len_shape(self):
        args = [
            [Key('a', 'A')],
            [Key('a', 'A'), Key('c', 'C')],
            [Key('a', 'A'), Key('b', 'B'), Key('c', 'C')],
        ]
        exps = [
            (1, (1, 2)),
            (2, (2, 2)),
            (3, (3, 2)),
        ]

        for a, (e1, e2) in zip(args, exps, strict=False):
            kl = KeyList(*a)
            assert len(kl) == e1
            assert kl.shape == e2

    def test_index(self):
        args = [
            [Key('a', 'A'), Key('c', 'C')],
            [Key('a', 'A'), Key('b', 'B'), Key('c', 'C')],
        ]

        for a in args:
            kl = KeyList(*a)
            idx = kl.to_index()

            assert isinstance(idx, pd.MultiIndex)
            assert len(a) == len(idx)
            assert idx.isin([tuple(arg.__dict__.values()) for arg in a]).any()

    def test_in_isin(self):
        base = [Key('a', 'A'), Key('b', 'B'), Key('c', 'C')]

        kl = KeyList(*base)

        assert base[0] in kl
        assert (kl.isin(base[:-1]) == [True, True, False]).all(), (
            f'got {kl.isin(base[:-1])}'
        )
        assert Key('c', 'A') not in kl

    def test_copy(self):
        base = [Key('a', 'A'), Key('b', 'B'), Key('c', 'C')]

        kl = KeyList(*base)
        cp = kl.copy()

        kl.keys = kl.keys.iloc[:0]

        assert kl != cp


@pytest.mark.smoke
class TestProblemVariables:
    def test_prob_var_init(self):
        args = [1, 5, 20, 100]
        for arg in args:
            ProblemVariables(arg)

    def test_prob_var(self):
        arg = 3
        var = ProblemVariables(arg)

        assert var.T.shape == (3, 3)
        assert var.p.shape == (3,)
        assert var.s.shape == (3,)
        assert var.sl_p.shape == (3,)
        assert var.sl_n.shape == (3,)

        fmt = var.format()
        assert str(var.T.value) in fmt
        assert str(var.p.value) in fmt
        assert str(var.s.value) in fmt
        assert str(var.sl_p.value) in fmt
        assert str(var.sl_n.value) in fmt
