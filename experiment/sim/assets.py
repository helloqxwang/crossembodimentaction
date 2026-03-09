from __future__ import annotations

import shutil
from pathlib import Path
from xml.etree import ElementTree


def _base_removal_targets(src: Path) -> set[str]:
    name = src.name.lower()
    if "shadow" in name:
        return {"forearm"}
    if "ezgripper" in name:
        return {"base_link"}
    return set()


def _joint_parent_child(joint_el: ElementTree.Element) -> tuple[str | None, str | None]:
    parent_el = joint_el.find("parent")
    child_el = joint_el.find("child")
    parent = parent_el.attrib.get("link") if parent_el is not None else None
    child = child_el.attrib.get("link") if child_el is not None else None
    return parent, child


def _remove_urdf_links(root: ElementTree.Element, remove_links: set[str]) -> None:
    if not remove_links:
        return

    changed = True
    while changed:
        changed = False
        for joint_el in list(root.findall("joint")):
            parent, child = _joint_parent_child(joint_el)
            if child not in remove_links:
                continue

            for other in list(root.findall("joint")):
                if other is joint_el:
                    continue
                other_parent_el = other.find("parent")
                if other_parent_el is not None and other_parent_el.attrib.get("link") == child and parent is not None:
                    other_parent_el.attrib["link"] = parent

            root.remove(joint_el)
            changed = True

    for joint_el in list(root.findall("joint")):
        parent, child = _joint_parent_child(joint_el)
        if parent in remove_links or child in remove_links:
            root.remove(joint_el)

    for link_el in list(root.findall("link")):
        if link_el.attrib.get("name") in remove_links:
            root.remove(link_el)


def _build_visual_scale_lookup(root: ElementTree.Element) -> tuple[dict[str, str], dict[str, str]]:
    by_name: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    for visual in root.findall(".//visual"):
        mesh = visual.find(".//mesh")
        if mesh is None:
            continue
        filename = mesh.attrib.get("filename", "")
        scale = mesh.attrib.get("scale", "")
        if not filename or not scale:
            continue
        basename = Path(filename).name
        stem = Path(filename).stem
        by_name[basename] = scale
        by_stem[stem] = scale
    return by_name, by_stem


def _find_existing_mesh_source(hand_dir: Path, original_rel: str) -> Path | None:
    target = (hand_dir / original_rel).resolve()
    if target.exists():
        return target

    basename = Path(original_rel).name
    stem = Path(original_rel).stem
    matches = list(hand_dir.rglob(basename))
    if not matches:
        matches = [path for path in hand_dir.rglob(f"{stem}.*") if path.suffix.lower() in {".obj", ".stl", ".dae"}]
    if not matches:
        return None
    return sorted(matches)[0]


def prepare_hand_path_for_mujoco(
    hand_path: str,
    *,
    patch_dir: Path,
    cache: dict[str, str],
) -> str:
    hand_path = str(Path(hand_path).expanduser().resolve())
    if hand_path in cache:
        return cache[hand_path]

    src = Path(hand_path)
    if src.suffix.lower() != ".urdf":
        cache[hand_path] = hand_path
        return hand_path

    tree = ElementTree.parse(src)
    root = tree.getroot()
    _remove_urdf_links(root, _base_removal_targets(src))
    hand_dir = src.parent
    visual_scale_by_name, visual_scale_by_stem = _build_visual_scale_lookup(root)

    robot_patch_dir = patch_dir / src.stem
    robot_patch_dir.mkdir(parents=True, exist_ok=True)
    patched_urdf_path = robot_patch_dir / src.name

    unresolved: list[str] = []
    mesh_files: list[tuple[str, Path]] = []
    for mesh in root.findall(".//mesh"):
        rel = mesh.attrib.get("filename", "")
        if not rel:
            continue
        source = _find_existing_mesh_source(hand_dir, rel)
        if source is None:
            unresolved.append(rel)
            continue
        mesh_files.append((rel, source))

    for rel, source in mesh_files:
        dst = (robot_patch_dir / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(source, dst)

        if "/visual/" in rel:
            rel_collision = rel.replace("/visual/", "/collision/")
            dst_collision = (robot_patch_dir / rel_collision).resolve()
            dst_collision.parent.mkdir(parents=True, exist_ok=True)
            if not dst_collision.exists():
                shutil.copy2(source, dst_collision)

    for collision in root.findall(".//collision"):
        mesh = collision.find(".//mesh")
        if mesh is None:
            continue
        if mesh.attrib.get("scale"):
            continue
        filename = mesh.attrib.get("filename", "")
        if not filename:
            continue
        basename = Path(filename).name
        stem = Path(filename).stem
        scale = visual_scale_by_name.get(basename) or visual_scale_by_stem.get(stem)
        if scale:
            mesh.set("scale", scale)

    if unresolved:
        cache[hand_path] = hand_path
        return hand_path

    tree.write(patched_urdf_path, encoding="utf-8", xml_declaration=True)
    cache[hand_path] = str(patched_urdf_path)
    return str(patched_urdf_path)
