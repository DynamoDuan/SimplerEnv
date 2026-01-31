import numpy as np
import trimesh
import sapien.core as sapien
from sapien.core import Actor


def normalize_vector(x, eps=1e-6):
    """Normalize a vector."""
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)


def get_component_mesh(actor: sapien.Actor, to_world_frame=True):
    """Get trimesh mesh from sapien actor."""
    meshes = []
    for shape in actor.get_collision_shapes():
        geometry = shape.geometry
        if isinstance(geometry, sapien.BoxGeometry):
            # Box geometry
            half_size = np.array(geometry.half_lengths)
            mesh = trimesh.creation.box(half_size * 2)
        elif isinstance(geometry, sapien.CapsuleGeometry):
            # Capsule geometry
            radius = geometry.radius
            half_length = geometry.half_length
            mesh = trimesh.creation.capsule(radius=radius, height=half_length * 2)
        elif isinstance(geometry, sapien.SphereGeometry):
            # Sphere geometry
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=geometry.radius)
        elif isinstance(geometry, sapien.PlaneGeometry):
            # Plane geometry - create a large box
            mesh = trimesh.creation.box([10, 10, 0.01])
        else:
            # Try to get vertices from geometry
            if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                vertices = geometry.vertices * geometry.scale
                if hasattr(geometry, 'indices') and geometry.indices is not None:
                    faces = geometry.indices.reshape(-1, 3)
                else:
                    # Try to infer faces
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(vertices)
                    faces = hull.simplices
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            else:
                continue
        
        # Apply local pose
        local_pose = shape.get_local_pose()
        if to_world_frame:
            world_pose = actor.get_pose() * local_pose
            transform = world_pose.to_transformation_matrix()
        else:
            transform = local_pose.to_transformation_matrix()
        
        mesh.apply_transform(transform)
        meshes.append(mesh)
    
    if not meshes:
        return None
    
    # Combine all meshes
    if len(meshes) == 1:
        return meshes[0]
    else:
        return trimesh.util.concatenate(meshes)


def get_actor_bbox(actor: Actor, to_world_frame=True, vis=False):
    """Get axis-aligned bounding box for an actor."""
    mesh = get_component_mesh(actor, to_world_frame=to_world_frame)
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    bbox = mesh.bounding_box.bounds

    return bbox


def get_actor_obb(actor: Actor, to_world_frame=True, vis=False):
    """Get oriented bounding box for an actor."""
    mesh = get_component_mesh(actor, to_world_frame=to_world_frame)
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented

    if vis:
        obb.visual.vertex_colors = (255, 0, 0, 10)
        trimesh.Scene([mesh, obb]).show()

    return obb


def compute_grasp_info_by_obb(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    # Assume normalized
    approaching = np.array(approaching)

    # Find the axis closest to approaching vector
    angles = approaching @ T[:3, :3]  # [3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]

    # Find the shorter axis as closing vector
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]

    # If sizes are close, choose the one closest to the target closing
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = T[:3, ind1]
        vec2 = T[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = T[:3, ind1]

    # Flip if far from target
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Reorder extents
    extents = extents[[ind0, ind1, ind2]]

    # Find the origin on the surface
    center = T[:3, 3].copy()
    half_size = extents[0] * 0.5
    center = center + approaching * (-half_size + min(depth, half_size))

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = normalize_vector(closing)

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info


