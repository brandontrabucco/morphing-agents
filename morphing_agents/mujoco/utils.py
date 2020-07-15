import os
import xml.etree.ElementTree as ET


def load_xml_tree(xml_path):
    """Utility function for loading mujoco xml files that may contain
    nested include tags

    Args:

    xml_path: str
        a path to a mujoco xml file with include tags to be expanded

    Returns:

    tree: ElementTree.Element
        an element that represents the root node of an xml tree
    """

    with open(xml_path, "r") as f:

        root = ET.fromstringlist(f.readlines()[16:])

        for c in root.findall(".//include"):
            file = c.attrib['file']
            target = os.path.join(os.path.dirname(xml_path), file)

            p = root.find(f".//include[@file='{file}']...")
            i = list(p).index(c)
            p.remove(c)

            for s in reversed(load_xml_tree(target)):
                p.insert(i, s)

        return root
