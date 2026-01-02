"""Generate random URDF kinematic chains.

Creates ``n`` URDF files, each describing a chain of ``k`` links connected by
revolute joints with random rotation axes. Links are randomly chosen as boxes
or cylinders with varied dimensions. The chain is arranged along each link's
local +X axis with a small clearance to avoid overlap.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union


CLEARANCE = 0.02  # gap between consecutive links (meters)
DEFAULT_NUM_CHAINS = 5
DEFAULT_NUM_LINKS = 4


@dataclass
class Box:
	size_x: float
	size_y: float
	size_z: float

	@property
	def length_along_x(self) -> float:
		return self.size_x


@dataclass
class Cylinder:
	radius: float
	length: float  # cylinder height along +X

	@property
	def length_along_x(self) -> float:
		return self.length


LinkShape = Union[Box, Cylinder]


@dataclass
class LinkSpec:
	name: str
	shape: LinkShape
	mass: float


@dataclass
class JointSpec:
	name: str
	parent: str
	child: str
	axis: tuple[float, float, float]
	origin_xyz: tuple[float, float, float]
	origin_rpy: tuple[float, float, float]


def random_box() -> Box:
	# Dimensions in meters; keep within a sensible range for stability.
	return Box(
		size_x=random.uniform(0.08, 0.25),
		size_y=random.uniform(0.04, 0.12),
		size_z=random.uniform(0.04, 0.12),
	)


def random_cylinder() -> Cylinder:
	return Cylinder(
		radius=random.uniform(0.02, 0.08),
		length=random.uniform(0.10, 0.30),
	)


def random_shape() -> LinkShape:
	return random.choice([random_box, random_cylinder])()


def normalized_random_axis() -> tuple[float, float, float]:
	while True:
		x, y, z = (random.uniform(-1.0, 1.0) for _ in range(3))
		norm = math.sqrt(x * x + y * y + z * z)
		if norm > 1e-6:
			return (x / norm, y / norm, z / norm)


def make_links(num_links: int) -> List[LinkSpec]:
	links: List[LinkSpec] = []
	for idx in range(num_links):
		shape = random_shape()
		mass = random.uniform(0.3, 2.0)
		links.append(LinkSpec(name=f"link_{idx}", shape=shape, mass=mass))
	return links


def make_joints(links: List[LinkSpec]) -> List[JointSpec]:
	joints: List[JointSpec] = []
	for idx in range(len(links) - 1):
		parent = links[idx]
		child = links[idx + 1]
		# Place child along parent's +X with clearance to avoid overlap.
		offset = (parent.shape.length_along_x / 2) + CLEARANCE + (
			child.shape.length_along_x / 2
		)
		axis = normalized_random_axis()
		joints.append(
			JointSpec(
				name=f"joint_{idx}",
				parent=parent.name,
				child=child.name,
				axis=axis,
				origin_xyz=(offset, 0.0, 0.0),
				origin_rpy=(0.0, 0.0, 0.0),
			)
		)
	return joints


def inertial_block(mass: float) -> str:
	# Simple diagonal inertia approximation; not physically exact but valid URDF.
	ixx = iyy = izz = mass * 0.01
	return (
		f"    <inertial>\n"
		f"      <mass value=\"{mass:.3f}\"/>\n"
		f"      <inertia ixx=\"{ixx:.4f}\" ixy=\"0\" ixz=\"0\" iyy=\"{iyy:.4f}\" iyz=\"0\" izz=\"{izz:.4f}\"/>\n"
		f"    </inertial>\n"
	)


def visual_block(shape: LinkShape) -> str:
	if isinstance(shape, Box):
		geometry = (
			f"      <box size=\"{shape.size_x:.3f} {shape.size_y:.3f} {shape.size_z:.3f}\"/>\n"
		)
	else:
		geometry = (
			f"      <cylinder radius=\"{shape.radius:.3f}\" length=\"{shape.length:.3f}\"/>\n"
		)
	return (
		f"    <visual>\n"
		f"      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
		f"      <geometry>\n"
		f"{geometry}"
		f"      </geometry>\n"
		f"      <material name=\"gray\"><color rgba=\"0.7 0.7 0.7 1\"/></material>\n"
		f"    </visual>\n"
	)


def collision_block(shape: LinkShape) -> str:
	if isinstance(shape, Box):
		geometry = (
			f"      <box size=\"{shape.size_x:.3f} {shape.size_y:.3f} {shape.size_z:.3f}\"/>\n"
		)
	else:
		geometry = (
			f"      <cylinder radius=\"{shape.radius:.3f}\" length=\"{shape.length:.3f}\"/>\n"
		)
	return (
		f"    <collision>\n"
		f"      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
		f"      <geometry>\n"
		f"{geometry}"
		f"      </geometry>\n"
		f"    </collision>\n"
	)


def link_block(link: LinkSpec) -> str:
	return (
		f"  <link name=\"{link.name}\">\n"
		f"{inertial_block(link.mass)}"
		f"{visual_block(link.shape)}"
		f"{collision_block(link.shape)}"
		f"  </link>\n"
	)


def joint_block(joint: JointSpec) -> str:
	ax_x, ax_y, ax_z = joint.axis
	ox, oy, oz = joint.origin_xyz
	rx, ry, rz = joint.origin_rpy
	return (
		f"  <joint name=\"{joint.name}\" type=\"revolute\">\n"
		f"    <parent link=\"{joint.parent}\"/>\n"
		f"    <child link=\"{joint.child}\"/>\n"
		f"    <origin xyz=\"{ox:.3f} {oy:.3f} {oz:.3f}\" rpy=\"{rx:.3f} {ry:.3f} {rz:.3f}\"/>\n"
		f"    <axis xyz=\"{ax_x:.4f} {ax_y:.4f} {ax_z:.4f}\"/>\n"
		f"    <limit lower=\"-3.1416\" upper=\"3.1416\" effort=\"5\" velocity=\"5\"/>\n"
		f"  </joint>\n"
	)


def robot_urdf(name: str, links: List[LinkSpec], joints: List[JointSpec]) -> str:
	blocks: List[str] = [f"<robot name=\"{name}\">\n"]
	for link in links:
		blocks.append(link_block(link))
	for joint in joints:
		blocks.append(joint_block(joint))
	blocks.append("</robot>\n")
	return "".join(blocks)


def generate_chain(index: int, num_links: int, out_dir: Path) -> Path:
	links = make_links(num_links)
	joints = make_joints(links)
	robot_name = f"chain_{index}"
	urdf_text = robot_urdf(robot_name, links, joints)
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f"{robot_name}.urdf"
	out_path.write_text(urdf_text)
	return out_path


def generate_multiple(count: int, num_links: int, out_dir: Path, seed: int | None) -> List[Path]:
	if seed is not None:
		random.seed(seed)
	paths = [generate_chain(i, num_links, out_dir) for i in range(count)]
	return paths


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate random URDF kinematic chains")
	parser.add_argument("n", type=int, nargs="?", default=DEFAULT_NUM_CHAINS, help="number of chains to generate")
	parser.add_argument("k", type=int, nargs="?", default=DEFAULT_NUM_LINKS, help="number of links per chain")
	parser.add_argument("--out", type=Path, default=Path("./out_chains"), help="output directory")
	parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
	return parser.parse_args(args=argv)


def main(argv: Iterable[str] | None = None) -> None:
	args = parse_args(argv)
	count = max(1, args.n)
	num_links = max(2, args.k)  # need at least two links for a joint
	paths = generate_multiple(count, num_links, args.out, args.seed)
	print(f"Generated {len(paths)} chains in {args.out.resolve()}")
	for path in paths:
		print(f" - {path}")


if __name__ == "__main__":
	main()
