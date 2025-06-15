import numpy as np
import numpy.typing as npt
import pandas as pd

from itaph.typing import KeyList


class Inventory:
    keys: KeyList  # Inventory key
    inventory: pd.DataFrame  # inventory dataframe

    def __init__(
        self,
        keys: KeyList,
        value: npt.NDArray[np.float64],
    ) -> None:
        self.keys = keys

        if len(self.keys) != len(value):
            raise ValueError(
                'Length not matched, '
                + f'key: {len(self.keys)} vs inventory: {len(value)}'
            )

        inventory = pd.DataFrame(value, index=self.keys.to_index())
        self.inventory = inventory.unstack('item')

    def __add__(self, inventory: 'Inventory') -> 'Inventory':
        self.keys.assert_eq(inventory.keys)
        ret = self.inventory.add(inventory.inventory, fill_value=0.0)
        return Inventory(self.keys, ret.stack('item'))

    def __sub__(self, inventory: 'Inventory') -> 'Inventory':
        self.keys.assert_eq(inventory.keys)
        ret = self.inventory.add(-inventory.inventory, fill_value=0.0)
        return Inventory(self.keys, ret.stack('item'))
