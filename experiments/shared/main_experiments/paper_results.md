# Results reported in the paper

**Qwen3.5-4B trained on CodeGym**, evaluated on four target environments without further training. Scores are percentages from [arXiv:2609.36116v1, Table 3](https://arxiv.org/pdf/2609.36116v1#page=6), with evaluations over three seeds.

| Method | ALFWorld | WebShop | τ²-bench | SWE-bench Verified |
|---|---:|---:|---:|---:|
| Base model | 23.45 | 28.54 | 61.05 | 43.21 |
| GRPO | 32.18 | 33.65 | 64.10 | 47.98 |
| Dyad-GRPO | **33.65** | **35.71** | **65.26** | 45.02 |
| GiGPO | 31.09 | 27.97 | 63.48 | 46.64 |
| Dyad-GiGPO | 31.72 | 31.21 | 64.78 | **49.61** |

These scores are reported in the paper. The current checkout has not been established as a reproduction of these published results.
