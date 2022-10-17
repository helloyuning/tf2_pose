
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
import scipy
import scipy.ndimage


def convert_to_homogenous(points):
    """
    Convert to homogenous coordinates
    Args:
        points: Input points
    Returns: Homogenized points
    """
    if points.shape[0] != 4:
        temp = np.ones((4, points.shape[1]))
        temp[:3, :] = points
        points = temp

    return points



def transform_points(points, transformation):
    """
    Apply transformation to the object points
    Args:
        points: Input points (non-homogeneous)
        transformation: Transformation matrix to apply
    Returns: transformed points
    """
    points = convert_to_homogenous(points)

    return np.dot(transformation, points)


def measure_add(model, gt_transformation, predicted_transformation):
    """
    Measure ADD metric
    Args:
        model: 3D object model
        gt_transformation: Ground truth transformation
        predicted_transformation: Predicted transformation
    Returns: ADD score
    """
    vertices_gt = transform_points(model.vertices.T, gt_transformation).T
    vertices_predicted = transform_points(model.vertices.T, predicted_transformation).T

    diff = vertices_gt - vertices_predicted
    diff = (diff * diff).sum(axis=-1)
    diff = np.mean(np.sqrt(diff))

    return diff / model.diameter_points