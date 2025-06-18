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
| $I^{[w]} + T^{[w]}(P^{[w]} + R^{[w-1]}) = I^{[w+1]} + R^{[w]} + (O^{[w]} - D^{[w]})$  | Flow balance for each $w \in \text{Week}$ |
| $T \geq 0$                                                                            | Non-negativity of transport plan          |
| $\text{Sum}(T, \text{axis} = [0, 1]) = 1$                                             | Total transport flow conservation         |
| $T \leq L_t$                                                                          | Transport capacity limits                 |
| $P \geq 0$                                                                            | Non-negativity of production plan         |
| $P \leq L_p$                                                                          | Production capacity limits                |
| $R + I \geq 0$                                                                        | No negative inventory levels              |
| $R + I \leq L_r$                                                                      | Inventory upper bounds                    |
| $D \geq 0$                                                                            | Non-negativity of shortage slack          |
| $O \geq 0$                                                                            | Non-negativity of overflow slack          |

---

### Variables

| Description                   | Name | Shape                                                                                               |
| ----------------------------- | ---- | --------------------------------------------------------------------------------------------------- |
| Transportation Plan           | $T$  | $(\text{item} \times \text{storage}) \times (\text{item} \times \text{storage}) \times \text{Week}$ |
| Production Plan               | $P$  | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |
| Retained Inventory Plan       | $R$  | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |
| Dropped Item Plan (shortages) | $D$  | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |
| Overflow Item Plan (excess)   | $O$  | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |

---

### Parameters

| Description                    | Name  | Shape                                                                                               |
| ------------------------------ | ----- | --------------------------------------------------------------------------------------------------- |
| Big-M penalty constant         | $M$   | Scalar                                                                                              |
| Sales Plan                     | $S$   | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |
| Inventory Plan                 | $I$   | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |
| Transportation Cost            | $C_t$ | $(\text{item} \times \text{storage}) \times (\text{item} \times \text{storage}) \times \text{Week}$ |
| Production Cost                | $C_p$ | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |
| Inventory Holding Cost         | $C_r$ | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |
| Transport Capacity Constraint  | $L_t$ | $(\text{item} \times \text{storage}) \times (\text{item} \times \text{storage}) \times \text{Week}$ |
| Production Capacity Constraint | $L_p$ | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |
| Inventory Capacity Constraint  | $L_r$ | $(\text{item} \times \text{storage}) \times \text{Week}$                                            |
