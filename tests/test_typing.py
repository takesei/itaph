import pytest

from itaph.typing import Key, KeyList, ProblemVariables


@pytest.mark.smoke
class TestKeyList:
    def test_key_list_init(self):
        args = [
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
                [Key('a', 'A'), Key('c', 'C')],
            ),
        ]

        exps = [True, False]

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
