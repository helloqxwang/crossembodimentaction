"""Generate random URDF kinematic chains.

Creates ``n`` URDF files, each describing a chain of ``k`` links connected by
revolute joints whose axes lie in the YZ plane. Links are boxes or capsules
laid out along +X with a small clearance between joints to avoid overlap.
"""

from __future__ import annotations

import numpy as np
import math
import random
import trimesh
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union


CLEARANCE = 0.0  # gap between consecutive links (meters)


@dataclass
class Box:
	size_x: float
	size_y: float
	size_z: float

	@property
	def length_along_x(self) -> float:
		return self.size_x
	
	@property
	def compact_info(self) -> float:
		return (0, self.size_x, self.size_y, self.size_z)


@dataclass
class Capsule:
	radius: float
	length: float  # cylindrical part along +X

	@property
	def length_along_x(self) -> float:
		# Total capsule extent includes two hemispheres.
		return self.length + 2 * self.radius
	
	@property
	def compact_info(self) -> float:
		# Total capsule extent includes two hemispheres.
		return (1, self.radius, self.length, 0)


LinkShape = Union[Box, Capsule]


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


def random_capsule() -> Capsule:
	return Capsule(
		radius=random.uniform(0.02, 0.08),
		length=random.uniform(0.10, 0.30),
	)


def random_shape() -> LinkShape:
	return random.choice([random_box, random_capsule])()


def fibonacci_sphere(samples: int, radius: float = 1.0) -> np.ndarray:
	"""Evenly spread points on a sphere surface using a Fibonacci lattice."""
	if samples <= 0:
		raise ValueError('samples must be positive.')
	indices = np.arange(samples, dtype=np.float64) + 0.5
	phi = (1.0 + math.sqrt(5.0)) / 2.0
	theta = 2.0 * math.pi * indices / phi
	z = 1.0 - 2.0 * indices / samples
	r_xy = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
	x = np.cos(theta) * r_xy
	y = np.sin(theta) * r_xy
	return radius * np.stack((x, y, z), axis=-1)


def link_to_trimesh(link: LinkShape) -> trimesh.Trimesh:
	"""Create a zero-centered trimesh for a Box or Capsule aligned to +X."""
	if isinstance(link, Box):
		return trimesh.creation.box(extents=(link.size_x, link.size_y, link.size_z))
	elif isinstance(link, Capsule):
		mesh = trimesh.creation.capsule(radius=link.radius, height=link.length)
		# trimesh capsule is along +Z; rotate to +X.
		rot = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
		mesh.apply_transform(rot)
		return mesh
	else:
		raise TypeError(f'Unsupported link type: {type(link)}')


def bps_from_link(
	link: LinkShape,
	num_points: int = 256,
	anchor_radius: float = 1.0,
) -> dict:
	"""Compute a simple BPS (Basis Point Set) for a zero-centered link.

	Returns anchors, distances, and scale factors so the user can rescale back.
	"""
	mesh = link_to_trimesh(link)
	# Scale to fit inside unit sphere for stable distances.
	half_extent = mesh.extents * 0.5
	max_radius = float(half_extent.max())
	scale_to_unit = 1.0 / max(max_radius, 1e-8)
	mesh.apply_scale(scale_to_unit)
	anchors = fibonacci_sphere(num_points, radius=anchor_radius)
	closest, _, _ = trimesh.proximity.closest_point(mesh, anchors)
	distances = np.linalg.norm(anchors - closest, axis=1)
	return {
		'anchors': anchors.astype(np.float32),
		'offsets': (anchors - closest).astype(np.float32),
		'distances': distances.astype(np.float32),
		'scale_to_unit': float(scale_to_unit),
		'scale_from_unit': float(1.0 / scale_to_unit),
		'shape_type': 'box' if isinstance(link, Box) else 'capsule',
	}


def normalized_random_axis_yz() -> tuple[float, float, float]:
	"""Sample a unit rotation axis constrained to the YZ plane (x = 0)."""
	while True:
		y, z = (random.uniform(-1.0, 1.0) for _ in range(2))
		norm = math.sqrt(y * y + z * z)
		if norm > 1e-6:
			return (0.0, y / norm, z / norm)


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
		# Place joint at the end of the parent link plus a small clearance.
		offset = parent.shape.length_along_x + CLEARANCE
		axis = normalized_random_axis_yz()
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


def inertial_block(mass: float, shape: LinkShape) -> str:
	# Simple diagonal inertia approximation; not physically exact but valid URDF.
	ixx = iyy = izz = mass * 0.01
	if isinstance(shape, Box):
		offset_x = shape.size_x / 2
	else:
		offset_x = shape.length_along_x / 2
	return (
		f"    <inertial>\n"
		f"      <origin xyz=\"{offset_x:.3f} 0 0\" rpy=\"0 0 0\"/>\n"
		f"      <mass value=\"{mass:.3f}\"/>\n"
		f"      <inertia ixx=\"{ixx:.4f}\" ixy=\"0\" ixz=\"0\" iyy=\"{iyy:.4f}\" iyz=\"0\" izz=\"{izz:.4f}\"/>\n"
		f"    </inertial>\n"
	)


def visual_block(shape: LinkShape) -> str:
	if isinstance(shape, Box):
		offset_x = shape.size_x / 2
		geometry = (
			f"      <box size=\"{shape.size_x:.3f} {shape.size_y:.3f} {shape.size_z:.3f}\"/>\n"
		)
	else:
		offset_x = shape.length_along_x / 2
		geometry = (
			f"      <capsule radius=\"{shape.radius:.3f}\" length=\"{shape.length:.3f}\"/>\n"
		)
	return (
		f"    <visual>\n"
		f"      <origin xyz=\"{offset_x:.3f} 0 0\" rpy=\"0 0 0\"/>\n"
		f"      <geometry>\n"
		f"{geometry}"
		f"      </geometry>\n"
		f"      <material name=\"gray\"><color rgba=\"0.7 0.7 0.7 1\"/></material>\n"
		f"    </visual>\n"
	)


def collision_block(shape: LinkShape) -> str:
	if isinstance(shape, Box):
		offset_x = shape.size_x / 2
		geometry = (
			f"      <box size=\"{shape.size_x:.3f} {shape.size_y:.3f} {shape.size_z:.3f}\"/>\n"
		)
	else:
		offset_x = shape.length_along_x / 2
		geometry = (
			f"      <capsule radius=\"{shape.radius:.3f}\" length=\"{shape.length:.3f}\"/>\n"
		)
	return (
		f"    <collision>\n"
		f"      <origin xyz=\"{offset_x:.3f} 0 0\" rpy=\"0 0 0\"/>\n"
		f"      <geometry>\n"
		f"{geometry}"
		f"      </geometry>\n"
		f"    </collision>\n"
	)


def link_block(link: LinkSpec) -> str:
	return (
		f"  <link name=\"{link.name}\">\n"
		f"{inertial_block(link.mass, link.shape)}"
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
	out_dir.mkdir(parents=True, exist_ok=True)
	robot_name = f"chain_{index}"

	links = make_links(num_links)
	joints = make_joints(links)
	
	bpses = [bps_from_link(link.shape) for link in links]
	links_property_np = np.stack([np.asarray(link.shape.compact_info) for link in links], axis=0)
	joints_property_np = np.stack([np.concatenate([joint.origin_xyz, joint.origin_rpy, joint.axis]).squeeze() for joint in joints], axis=0)
	np.savez(
		out_dir / f"{robot_name}_properties.npz",
		links_property=links_property_np,
		joints_property=joints_property_np,
		bpses=bpses,
	)
	
	urdf_text = robot_urdf(robot_name, links, joints)
	out_path = out_dir / f"{robot_name}.urdf"
	out_path.write_text(urdf_text)
	return out_path


def generate_multiple(count: int, min_len: int, max_len: int, out_dir: Path) -> List[Path]:
	paths = [generate_chain(i, random.randint(min_len, max_len), out_dir) for i in range(count)]
	return paths

if __name__ == "__main__":
	store_dir = "./data/out_chains"
	seed = 42
	min_len = 2
	max_len = 5
	num_chains = int(1e3 + 1e2)

	random.seed(seed)
	paths = generate_multiple(num_chains, min_len, max_len, Path(store_dir))
	print(f"Generated {len(paths)} chains in {Path(store_dir).resolve()}")
	for path in paths:
		print(f" - {path}")
