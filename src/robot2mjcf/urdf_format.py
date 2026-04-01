"""Normalize URDF XML formatting in place."""

import argparse
import shutil
import xml.dom.minidom as minidom
from pathlib import Path


def format_urdf_file(urdf_path: str | Path) -> None:
    urdf_path = Path(urdf_path)
    tmp_urdf_path = urdf_path.with_name(f"{urdf_path.stem}_tmp{urdf_path.suffix}")

    if tmp_urdf_path.exists():
        tmp_urdf_path.unlink()

    xml_doc = minidom.parse(str(urdf_path))

    with tmp_urdf_path.open("w") as fp:
        xml_doc.writexml(fp)

    shutil.move(tmp_urdf_path, urdf_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf_path", type=str)
    args = parser.parse_args()
    format_urdf_file(args.urdf_path)


if __name__ == "__main__":
    main()
