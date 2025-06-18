# Itaph
- author: Sei Takeda

## Optimization Problem Definition for Supply Chain Planning

This project supports a cost-minimization problem for production, transportation, and inventory planning over a finite time horizon.
The model accounts for production limits, transport constraints, and dynamic inventory behavior across multiple storage sites and items.
The aim is to fulfill a sales plan while avoiding shortages and minimizing excess inventory.
Slack variables (shortage and overflow) are penalized via Big-M constants, while temporal flow is managed through inventory dynamics and inter-week transport plans.
The formulation separates operational decisions (production, transport, inventory) from structural constraints and cost considerations, allowing scalable adaptation to multi-week planning horizons.

---

### Problem Formulation

**Minimize:**

$$
\text{Sum}(\langle C_t, T \rangle + \langle C_p, P \rangle + \langle C_r, R \rangle + M \cdot D + M \cdot O)
$$

Where:

* $\langle A, B \rangle$: Hadamard (element-wise) product
* $M$: Big-M constant to penalize slack variables

**Subject to:**

| Expression                                                                            | Description                               |
| ------------------------------------------------------------------------------------- | ----------------------------------------- |
| $T^{[w]}(I^{[w]} + P^{[w]} + R^{[w-1]}) = I^{[w+1]} + R^{[w]} + (O^{[w]} - D^{[w]})$  | Flow balance for each $w \in \text{Week}$ |
| $T \geq 0$                                                                            | Non-negativity of transport plan          |
| $\text{Sum}(T, \text{axis} = [0, 1, 4]) = 1$                                          | Total transport flow conservation         |
| $T \leq L_t$                                                                          | Transport capacity limits                 |
| $P \geq 0$                                                                            | Non-negativity of production plan         |
| $P \leq L_p$                                                                          | Production capacity limits                |
| $R + I \geq 0$                                                                        | No negative inventory levels              |
| $R + I \leq L_r$                                                                      | Inventory upper bounds                    |
| $D \geq 0$                                                                            | Non-negativity of shortage slack          |
| $O \geq 0$                                                                            | Non-negativity of overflow slack          |

---

### Variables

| Expr | Description                   | Shape                                                |
| ---- | ----------------------------- | ---------------------------------------------------- |
| $T$  | Transportation Plan           | (`item` × `storage`) × (`item` × `storage`) × `Week` |
| $P$  | Production Plan               | (`item` × `storage`) × `Week`                        |
| $R$  | Retained Inventory Plan       | (`item` × `storage`) × `Week`                        |
| $D$  | Dropped Item Plan (shortages) | (`item` × `storage`) × `Week`                        |
| $O$  | Overflow Item Plan (excess)   | (`item` × `storage`) × `Week`                        |

---

### Parameters

| Expr  | Description                    | Shape                                                |
| ----- | ------------------------------ | ---------------------------------------------------- |
| $M$   | Big-M penalty constant         | Scalar                                               |
| $S$   | Sales Plan                     | (`item` × `storage`) × `Week`                        |
| $I$   | Inventory Plan                 | (`item` × `storage`) × `Week`                        |
| $C_t$ | Transportation Cost            | (`item` × `storage`) × (`item` × `storage`) × `Week` |
| $C_p$ | Production Cost                | (`item` × `storage`) × `Week`                        |
| $C_r$ | Inventory Holding Cost         | (`item` × `storage`) × `Week`                        |
| $L_t$ | Transport Capacity Constraint  | (`item` × `storage`) × (`item` × `storage`) × `Week` |
| $L_p$ | Production Capacity Constraint | (`item` × `storage`) × `Week`                        |
| $L_r$ | Inventory Capacity Constraint  | (`item` × `storage`) × `Week`                        |
