import numpy as np
import numpy.typing as npt
import pandas as pd

from itaph.inventory import Inventory
from itaph.typing import Key, KeyList


class Operator:
    keys: KeyList
    params: pd.DataFrame

    def __init__(
        self,
        keys: KeyList,
        value: npt.NDArray[np.float64],
    ) -> None:
        if value.shape[0] != value.shape[1]:
            raise ValueError(
                f'Value must be an affine (square) matrix, got {value.shape}'
            )

        params = np.asarray(value)
        if len(keys) == len(params):  # value is an affine matrix
            self.keys = keys
        elif len(keys) + 1 == len(params):
            self.keys = keys.add(Key('_affine', '_affine'))
        else:
            raise ValueError(
                'Invalid length of keys,'
                + f' got {len(keys)}, expects {len(value)}(+1)'
            )

        self.params = pd.DataFrame(
            params,
            columns=self.keys.to_index().rename(['store_from', 'item_from']),
            index=self.keys.to_index().rename(['store_to', 'item_to']),
        )

    def __mul__(self, operator: 'Operator') -> 'Operator':
        self.keys.assert_eq(operator.keys)
        ret = self.params @ operator.params
        return Operator(self.keys, ret)

    def __matmul__(self, inventory: Inventory) -> Inventory:
        # print(self.keys.loc[(self.keys != '_affine').all(axis=0)])
        # self.keys.assert_eq(inventory.keys)
        vec = np.concatenate([inventory.vector, [1]])
        ret = self.params @ vec
        return Inventory(inventory.keys, ret[:-1])
