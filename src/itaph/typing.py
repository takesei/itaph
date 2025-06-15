from dataclasses import dataclass, field

import cvxpy as cp
import pandas as pd


@dataclass(frozen=True)
class Key:
    location: str
    item: str


class KeyList:
    def __init__(self, *keys: Key) -> None:
        self.keys = pd.DataFrame([k.__dict__ for k in keys])
        self.keys = self.keys.sort_values(list(self.keys.columns))

    def __eq__(self, key_list: 'KeyList') -> bool:
        is_same_fields = (self.keys.columns == key_list.keys.columns).all()
        is_same_length = len(self.keys) == len(key_list.keys)
        is_same_contents = self.keys.equals(key_list.keys)

        return is_same_fields and is_same_length and is_same_contents

    def assert_eq(self, key_list: 'KeyList') -> None:
        is_same_fields = (self.keys.columns == key_list.keys.columns).all()
        is_same_length = len(self.keys) == len(key_list.keys)
        is_same_contents = self.keys.equals(key_list.keys)

        if not (is_same_fields and is_same_length and is_same_contents):
            raise AssertionError(
                'KeyList not matched; '
                + f'{is_same_fields=}, {is_same_length=}, {is_same_contents=}'
            )

    def to_index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_frame(self.keys)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.shape

    def __len__(self) -> int:
        return len(self.keys)


@dataclass(frozen=True)
class ProblemVariables:
    N: int
    T: cp.Variable = field(init=False)
    p: cp.Variable = field(init=False)
    s: cp.Variable = field(init=False)
    sl_p: cp.Variable = field(init=False)
    sl_n: cp.Variable = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'T', cp.Variable((self.N, self.N), nonneg=True))
        object.__setattr__(self, 'p', cp.Variable(self.N, nonneg=True))
        object.__setattr__(self, 's', cp.Variable(self.N, nonneg=True))
        object.__setattr__(
            self, 'sl_p', cp.Variable(self.N, nonneg=True)
        )  # positive slack value
        object.__setattr__(
            self, 'sl_n', cp.Variable(self.N, nonneg=True)
        )  # negative slack value

    def format(self) -> str:
        return (
            'Tx + (p - s) + (slack_+ + slack_-) = \n'
            + f'{self.T.value}x + ({self.p.value} - {self.s.value}) + '
            + f'({self.sl_p.value} + {self.sl_n.value})'
        )

    def print(self) -> None:
        print(self.format())
