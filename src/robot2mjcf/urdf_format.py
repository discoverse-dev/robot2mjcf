import argparse
import os
import shutil
import xml.dom.minidom as minidom

parser = argparse.ArgumentParser()
parser.add_argument("urdf_path", type=str)
args = parser.parse_args()

urdf_path = args.urdf_path
tmp_urdf_path = urdf_path.replace(".urdf", "_tmp.urdf")

if os.path.exists(tmp_urdf_path):
    shutil.rmtree(tmp_urdf_path)

xmlDoc = minidom.parse(urdf_path)

with open(tmp_urdf_path, "w") as fp:
    xmlDoc.writexml(fp)

shutil.move(tmp_urdf_path, urdf_path)
