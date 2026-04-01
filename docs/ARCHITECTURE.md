# Architecture

## Scope

`robot2mjcf` converts URDF robot descriptions into MuJoCo MJCF and can optionally run XML/mesh post-processing steps to make the result easier to simulate.

## Main Flow

1. Parse the input URDF and metadata.
2. Build the MJCF tree and collect mesh/material assets.
3. Copy mesh resources into the output layout.
4. Save the initial MJCF document.
5. Run optional post-processing stages.

The top-level entry point is [`convert.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/convert.py).

## Module Responsibilities

- [`src/robot2mjcf/convert.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/convert.py): library/CLI entry point and orchestration.
- [`src/robot2mjcf/conversion_helpers.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_helpers.py): pure helpers extracted from the converter.
- [`src/robot2mjcf/conversion_postprocess.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_postprocess.py): central post-process orchestration and side-effect gating.
- [`src/robot2mjcf/mjcf_builders.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/mjcf_builders.py): MJCF subtree construction helpers.
- [`src/robot2mjcf/model.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/model.py): metadata schema.
- [`src/robot2mjcf/postprocess/`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/postprocess): XML and mesh post-processing modules.

## Post-Process Classes

There are two important categories:

- Pure XML transforms.
- XML + filesystem transforms that also rewrite mesh assets.

The second class is heavier, slower, and more environment-sensitive.

## Current Limits

- [`convert.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/convert.py) is still the orchestration hotspot.
- The mesh pipeline is still file-centric.
- Heavy post-processing is not yet split into optional extras.

## Recommended Extension Points

- Add new metadata schema in [`model.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/model.py).
- Add pure helpers in [`conversion_helpers.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_helpers.py) instead of growing `main()`.
- Add new stages behind [`conversion_postprocess.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_postprocess.py) so side effects stay centrally controlled.
- Add regression tests against real examples whenever output semantics change.
