import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # TODO
  # Normalize coordinates (to points on the normalized image plane)

  # K inverse
  K_inverse = np.zeros((K.shape))
  K_inverse[0,0] = 1/K[0,0]
  K_inverse[1,1] = 1/K[1,1]
  K_inverse[0,2] = -(K[0,2]/K[0,0])
  K_inverse[1,2] = -(K[1,2]/K[1,1])
  K_inverse[2,2] = 1.0
  
  # Normalize img1 keypoints: x̂ = K_inverse*x
  normalized_kps1 = []
  for keypoint in im1.kps:
    keypoint = np.append(keypoint,1.0)
    res = np.matmul(K_inverse,keypoint.transpose())
    normalized_kps1.append([res[0], res[1], res[2]])
  normalized_kps1 = np.asarray(normalized_kps1)

  # Normalize img2 keypoints: x̂ = K_inverse*x
  normalized_kps2 = []
  for keypoint in im2.kps:
    keypoint = np.append(keypoint,1.0)
    res = np.matmul(K_inverse,keypoint.transpose())
    normalized_kps2.append([res[0], res[1], res[2]])
  normalized_kps2 = np.asarray(normalized_kps2)
  
  # Assemble constraint matrix as equation 2.1
  constraint_matrix = np.zeros((matches.shape[0], 9))
  for i in range(matches.shape[0]):
    # TODO
    # Add the constraints
    x = normalized_kps1[matches[i,0],:][0]
    y = normalized_kps1[matches[i,0],:][1]
    x_ = normalized_kps2[matches[i,1],:][0]
    y_ = normalized_kps2[matches[i,1],:][1]

    constraint_matrix[i,0] = x_*x
    constraint_matrix[i,1] = x_*y
    constraint_matrix[i,2] = x_
    constraint_matrix[i,3] = y_*x
    constraint_matrix[i,4] = y_*y
    constraint_matrix[i,5] = y_
    constraint_matrix[i,6] = x
    constraint_matrix[i,7] = y
    constraint_matrix[i,8] = 1.0
    
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  # TODO
  # Reshape the vectorized matrix to it's proper shape again
  E_hat = vectorized_E_hat.reshape((3,3))

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  u, _, vh = np.linalg.svd(E_hat)
  s = [1,1,0]
  E = np.matmul(u*s, vh)

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]
    assert(abs(kp2.transpose() @ E @ kp1) < 0.01)

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

  Rt1 = np.append(R1, np.expand_dims(t1, 1), 1)
  Rt2 = np.append(R2, np.expand_dims(t2, 1), 1)

  # Filter points behind the first camera
  indexes1 = []
  for i,point in enumerate(points3D):
    # transfom 3d point into camera space by applying [R|t]
    point_cam1 = Rt1@MakeHomogeneous(point)
    # point lies behind camera when depth is negative
    if point_cam1[2] < 0:
        indexes1.append(i)
  points3D = np.delete(points3D, indexes1, axis=0) # axis 0 deltets row
  im1_corrs = np.delete(im1_corrs,indexes1, axis=0)
  
  # Filter points behind the second camera
  indexes2 = []
  for i,point in enumerate(points3D):
    # transfom 3d point into camera space by applying [R|t]
    point_cam2 = Rt2@MakeHomogeneous(point)
    # point lies behind camera when depth is negative
    if point_cam2[2] < 0:
        indexes2.append(2)
  im2_corrs = np.delete(im2_corrs,indexes2,axis=0)
  points3D = np.delete(points3D, indexes2, axis=0) # axis 0 deltets row

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  # K inverse
  K_inverse = np.zeros((K.shape))
  K_inverse[0,0] = 1/K[0,0]
  K_inverse[1,1] = 1/K[1,1]
  K_inverse[0,2] = -(K[0,2]/K[0,0])
  K_inverse[1,2] = -(K[1,2]/K[1,1])
  K_inverse[2,2] = 1.0

  normalized_points2D = [K_inverse@MakeHomogeneous(point) for point in points2D]
  normalized_points2D = np.asarray(normalized_points2D)

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images
  image = images[image_name] # new image
  points3D = np.zeros((0,3))

  corrs = {}
  for name in registered_images:
    reg_image = images[name]
    e_matches = GetPairMatches(image_name, name, matches)
    points3D, im1_corrs, im2_corrs = TriangulatePoints(K, image, reg_image, e_matches)
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
    corrs[image_name] = im1_corrs
    corrs[name] = im2_corrs
  return points3D, corrs
  
