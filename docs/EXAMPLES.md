# Example Walkthrough

This repository ships with two real robot examples:

- [`examples/agilex-piper`](/Users/jiayufei/ws/robot2mjcf/examples/agilex-piper)
- [`examples/realman-rm65`](/Users/jiayufei/ws/robot2mjcf/examples/realman-rm65)

## Quick Start

### agilex-piper

```bash
cd examples/agilex-piper
robot2mjcf piper.urdf \
  -o output_mjcf/piper.xml \
  -m metadata/metadata.json \
  -am metadata/actuator.json \
  -dm metadata/default.json \
  -a metadata/appendix.xml
```

### realman-rm65

```bash
cd examples/realman-rm65
robot2mjcf rm65b_eg24c2_description.urdf \
  -o output_mjcf/rm65.xml \
  -m metadata/metadata.json \
  -am metadata/actuator.json \
  -dm metadata/default.json \
  -a metadata/appendix.xml
```

## Refactor Regression Method

Use the real examples as the baseline.

The repository regression tests in [`tests/test_convert.py`](/Users/jiayufei/ws/robot2mjcf/tests/test_convert.py) compare semantic output instead of raw XML bytes:

- model name
- body names
- joint names
- actuator names
- equality pairs
- XML counts
- referenced mesh existence
- MuJoCo loadability and model counts

This is the preferred refactor safety net.
