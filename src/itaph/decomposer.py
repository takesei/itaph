import cvxpy as cp
import numpy as np
import pandas as pd

from itaph.inventory import Inventory
from itaph.typing import KeyList, ProblemVariables


class Decomposer:
    """
    Inv_out + (slack_pos + slack_neg) = T @ Inv_in + (p - s)
    number of variables
    - inv_in, inv_out: N
    → T + p + s + slack_pos + slack_neg ... N^2 + N + N + N + N = N^2 + 4N
    N = n_storage_loactions x n_items
    """

    def __init__(
        self,
        keys: KeyList,
        cost_tran: pd.DataFrame,
        cost_prod: pd.DataFrame,
        cost_stre: pd.DataFrame,
        *,
        big_m: float = 1e6,
    ) -> None:
        self.keys = keys
        self.N = len(self.keys)
        self.M = big_m  # slack penalty

        self._assert_valid_cost(cost_tran, cost_prod, cost_stre)

        try:
            self.C_t = np.asarray(
                cost_tran.loc[self.keys.to_index(), self.keys.to_index()]
            )
            self.c_p = np.asarray(cost_prod.loc[self.keys.to_index()]).flatten()
            self.c_s = np.asarray(cost_stre.loc[self.keys.to_index()]).flatten()
        except KeyError as e:
            cond_tran = cost_tran.index[~cost_tran.index.isin(self.keys.to_index())]
            cond_prod = cost_prod.index[~cost_prod.index.isin(self.keys.to_index())]
            cond_stre = cost_stre.index[~cost_stre.index.isin(self.keys.to_index())]
            raise KeyError(
                'Arg <keys> must be used in all dataframes, '
                + f'cost_tran: {cond_tran}'
                + f'cost_prod: {cond_prod}'
                + f'cost_stre: {cond_stre}'
            ) from e

    def decompose(
        self, inp: Inventory, out: Inventory, *, verbose: bool = False
    ) -> ProblemVariables:
        self._assert_valid_io(inp, out)

        # Def params
        i_in = np.asarray(inp.vector)
        i_out = np.asarray(out.vector)

        # Def Lp vars
        var = ProblemVariables(self.N)
        T = var.T  # noqa: N806
        p = var.p
        s = var.s
        sl_p = var.sl_p
        sl_n = var.sl_n

        # Def problem
        objective = cp.Minimize(
            cp.sum(
                cp.multiply(self.C_t, T)
                + self.c_p @ p
                + self.c_s @ s
                + self.M * (cp.sum(sl_p) + cp.sum(sl_n))
            )
        )

        constraints = [
            # Not exact, p should be T@p (violates DCP)
            T @ i_in + p == i_out + s + sl_p - sl_n,
            # mass conservation
            cp.sum(T @ i_in) == np.sum(i_in),
        ]

        prob = cp.Problem(objective, constraints)

        # Solve
        prob.solve(solver=cp.HIGHS, verbose=verbose)

        if prob.status != 'optimal':
            raise RuntimeError(f'Solver status is not optimal, got {prob}')

        return var

    def _assert_valid_cost(
        self, cost_tran: pd.DataFrame, cost_prod: pd.DataFrame, cost_stre: pd.DataFrame
    ) -> None:
        is_square_transp = cost_tran.shape[0] == cost_tran.shape[1]
        is_same_len_transp = self.N == len(cost_tran)
        is_same_len_prod = self.N == len(cost_prod)
        is_same_len_store = self.N == len(cost_stre)
        if not (
            is_square_transp
            and is_same_len_transp
            and is_same_len_prod
            and is_same_len_store
        ):
            raise ValueError(
                'Cost params size not matched, '
                + f'{is_square_transp=}{is_same_len_transp=}, '
                + f'{is_same_len_prod=}, {is_same_len_store=}'
            )

    def _assert_valid_io(
        self,
        inp: Inventory,
        out: Inventory,
    ) -> None:
        is_same_idx_inp = self.keys == inp.keys
        is_same_idx_out = self.keys == out.keys
        if not (is_same_idx_inp and is_same_idx_out):
            raise ValueError(
                'Inventory index is not the same with Decomposer.keys; '
                + f'{is_same_idx_inp=}, {is_same_idx_out=}'
            )
