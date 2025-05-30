# 📘 Kernel Documentation: radigen.solve

## 🔧 Function
`radigen.solve(mixture, temp, oxygen, time)`

## 📥 Inputs
| Name     | Type    | Description                              |
|----------|---------|------------------------------------------|
| mixture  | str     | Name or identifier of chemical mixture    |
| temp     | float   | Temperature in Celsius                    |
| oxygen   | float   | Oxygen fraction (e.g., 0.21 for air)      |
| time     | float   | Duration in hours                         |

## 📤 Outputs
| Name                | Type      | Description                                |
|---------------------|-----------|--------------------------------------------|
| concentration_curves| dict/list | Time evolution of species concentrations   |
| radical_fluxes      | dict/list | Flux or generation rate of radical species |

## 🧠 Assumptions
- The model uses ordinary differential equations with mass-action kinetics.
- Radical generation from oxygen uptake is temperature-dependent.
- No external inhibitors or antioxidants are present unless specified in mixture.

## ⚠️ Limitations
- No spatial heterogeneity (0D model).
- Requires predefined mixture species and reactivity patterns.
- Output interpretation depends on naming conventions in `species.py`.

## 🔗 See Also
- `mixtureKinetics` class in `radigen`
- `TKO2cycle` if dynamic temperature/oxygen profiles are required

---

> Document updated: 2025-05-29

