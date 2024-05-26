import numpy as np
import trimesh
import igl

def to_trimesh(mesh):
    tri = trimesh.Trimesh(mesh.v, mesh.f, process=False)
    return tri

def compute_contact_labels(smpl, obj, human_joints_indices, num_samples, gamma=0.12):
    """
    Compute contact labels for each of the specified human body joints against an object using signed distance.
    
    :param smpl: trimesh.Trimesh, a mesh model of the human
    :param obj: trimesh.Trimesh, a mesh model of the object
    :param human_joints_indices: list of indices for primary joints in the SMPL model
    :param num_samples: number of points to sample on the object surface
    :param gamma: float, the distance threshold for determining contact
    :return: tuple (object_contact_points, contact_label)
        object_contact_points - points on the object surface that are in contact
        contact_label - 8-dimensional binary vector indicating contact status for each joint
    """
    object_points = obj.sample(num_samples)
    
    object_contact_points = []
    contact_label = [0] * len(human_joints_indices)
    
    # Calculate the signed distance from object points to the SMPL mesh
    dist, _, closest_points = igl.signed_distance(object_points, smpl.vertices, smpl.faces, return_normals=False)
    
    # Check each joint if it is in contact with any of the closest points
    for i, joint_index in enumerate(human_joints_indices):
        joint_point = smpl.vertices[joint_index]
        distances = np.linalg.norm(closest_points - joint_point, axis=1)

        min_dist_index = np.argmin(distances)
        if distances[min_dist_index] < gamma:
            contact_label[i] = 1
            object_contact_points.append(closest_points[min_dist_index])
        else:
            object_contact_points.append(None)  # No contact point close enough

    return object_contact_points, contact_label

# Example usage:
smpl_model = trimesh.load_mesh('data/objects/smpl.obj')
obj_model = trimesh.load_mesh('data/objects/cup.obj')
joint_indices = [0, 1, 2, 3, 4, 5, 6, 7]
# with hand_model joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ... , 50]
# 8 + 21 + 21 = 50

contact_points, contact_status = compute_contact_labels(to_trimesh(smpl_model), to_trimesh(obj_model), joint_indices, 512)

