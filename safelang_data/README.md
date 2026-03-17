# SafeLang-1M Dataset

**Unified safety benchmark corpus** aggregating 10 public datasets for LLM safety research.

## Stats

| Split | Size |
|-------|------|
| Train | ~944k |
| Validation | ~118k |
| Test | ~118k |
| **Total** | **~1.18M** |

## Sources

| Source | HF Path | Label |
|--------|---------|-------|
| ToxiGen | `skg/toxigen-data` | Toxic ≥ 0.5 → unsafe |
| RealToxicityPrompts | `allenai/real-toxicity-prompts` | `challenging` |
| AdvBench | `walledai/AdvBench` | All unsafe |
| JailbreakBench | `JailbreakBench/JBB-Behaviors` | All unsafe |
| XSTest | `re-align/just-eval-instruct` | All safe (false-positive probes) |
| BeaverTails | `PKU-Alignment/BeaverTails` | `is_safe` inverted |
| WildGuard | `allenai/wildguardmix` | `prompt_harm_label` == harmful |
| SafeNLP | `Anthropic/hh-rlhf` | RLHF chosen = safe |
| PromptBench | `microsoft/promptbench` | `label` |
| HarmBench | `walledai/HarmBench` | All unsafe |

## Schema

```json
{"text": "...", "label": 0, "source": "beavertails"}
```

- `label`: `0` = safe, `1` = unsafe
- `source`: benchmark origin

## Reproducibility

Build from scratch:
```bash
python datasets/safelang_1m.py --seed 42
```

SHA-256 checksums per split are printed and saved to `stats.json`.

## License

All source datasets are public and used under their respective licenses.
The unified SafeLang-1M corpus is released under CC BY 4.0.
