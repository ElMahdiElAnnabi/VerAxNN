# qmacverify

A self-contained framework repo for exporting quantized INT8 CNN graphs, running exact and approximate integer inference, verifying approximate operators across Python/Verilog, and extracting sound per-layer constraints using interval bound propagation (IBP).

## Pipeline Overview

1. **Export** a quantized graph + parameters from a model definition.
2. **Exact integer runner** for deterministic, bit-accurate INT8/INT32 inference.
3. **Approximate ops** (Python + Verilog) for truncated multiply/add.
4. **Opcheck** to cross-check Python vs Verilog operator truth tables.
5. **Constraints** extraction (sound bounds) via IBP for each conv layer.

Sound vs tight: **sound** bounds are conservative and guaranteed to contain all possible values; **tight** bounds are empirical (optional) and may be smaller but are not guaranteed.

## Quickstart (Demo)

```bash
python -m qmacverify.experiments.demo_pipeline --model cnn_small --drop 2 --num-opcheck 2000
```

## Key CLI Commands

### Export
```bash
python -m qmacverify.export.export_quantized --model cnn_small --out export/cnn_small
```

### Exact run
```bash
python -m qmacverify.runner.run_exact_int --pkg export/cnn_small --random-input --out results/baseline/cnn_small
```

### Approx run
```bash
python -m qmacverify.runner.run_approx_int --pkg export/cnn_small --amul trunc --amul-drop 2 --aadd trunc --aadd-drop 2 --random-input --out results/approx/cnn_small
```

### Constraints extraction
```bash
python -m qmacverify.constraints.extract_D --pkg export/cnn_small --out constraints/cnn_small/D_sound.json
```

### Opcheck (Python vs Verilog)
```bash
python -m qmacverify.opcheck.opcheck_cli --drop 2 --num 5000 --out results/opcheck
```

## Outputs

- `export/<model>/graph.json`, `params.npz`, `quant_spec.json`
- `results/baseline/<model>/outputs.npz`
- `results/approx/<model>/outputs.npz`
- `results/compare/<model>/diff_report.json`
- `constraints/<model>/D_sound.json`
- `results/opcheck/*_equiv_report.json`

## Notes

- Random seeds are fixed for reproducibility.
- If `iverilog` is not installed, Verilog simulations are skipped with a friendly message.
- All paths are relative to the repo root.
