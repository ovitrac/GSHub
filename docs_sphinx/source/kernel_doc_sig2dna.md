# 📘 Kernel Documentation: sig2dna.encode

## 🔧 Function
`sig2dna.encode(signal, scales)`

## 📥 Inputs
| Name    | Type      | Description                                          |
|---------|-----------|------------------------------------------------------|
| signal  | array/list| 1D analytical signal (e.g., GC-MS trace)             |
| scales  | list[int] | List of wavelet scales to analyze (e.g., [1, 2, 4])  |

## 📤 Outputs
| Name         | Type    | Description                                         |
|--------------|---------|-----------------------------------------------------|
| dna_sequence | str     | Encoded symbolic sequence (e.g., YAZBZZ...)         |
| motifs       | list    | Detected symbolic motifs or features                |

## 🧠 Assumptions
- Signal is pre-processed and normalized.
- CWT-based encoding using Ricker or Gaussian wavelets.
- Motif extraction is based on thresholded energy and position features.

## ⚠️ Limitations
- Encoding is sensitive to noise and baseline drift.
- Interpretability depends on motif dictionary and resolution.
- Sequence length depends on signal resolution and selected scales.

## 🔗 See Also
- `DNAsignal` and `DNAstr` classes for decoding and alignment
- `signal_collection.generate_synthetic()` for testing and benchmarking

---

> Document updated: 2025-05-29

