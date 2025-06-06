# 📘 Kernel Documentation: sfppy.evaluate

## 🔧 Function
`sfppy.evaluate(material, food, temperature, duration)`

## 📥 Inputs
| Name        | Type    | Description                                      |
|-------------|---------|--------------------------------------------------|
| material    | str     | Name of the food contact material                |
| food        | str     | Type of food or simulant (e.g., oliveoil, ethanol) |
| temperature | float   | Exposure temperature in Celsius                  |
| duration    | float   | Contact time in hours                            |

## 📤 Outputs
| Name             | Type    | Description                                     |
|------------------|---------|-------------------------------------------------|
| migration_profile| dict    | Estimated migration vs. time and conditions     |
| compliance_status| bool    | True if migration complies with EU regulations  |

## 🧠 Assumptions
- Uses partition/diffusion models calibrated on EU regulation.
- Assumes homogeneous contact surface and fixed geometry.
- Temperature influences both diffusion coefficient and partitioning.

## ⚠️ Limitations
- Only models monolayer materials (use separate model for multilayers).
- Does not account for degradation or chemical reaction during migration.
- Food simulant must match a validated list.

## 🔗 See Also
- SFPPy `material_db` and `migration_solver`
- CosPaTox dataset integration for extended scenarios

---

> Document updated: 2025-05-29

