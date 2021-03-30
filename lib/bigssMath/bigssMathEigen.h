#ifndef _BIGSS_MATH_EIGEN_H
#define _BIGSS_MATH_EIGEN_H

#include <eigenAddons.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace BIGSS {

  /// Solves the AX=XB problem on the Euclidean group SE(3). 
  /// Based off the Matlab implementation by Dr. Yoshito Otake, following the
  /// algorithm described by: 
  ///   Park FC and Martin BJ, Robot sensor calibration: Solving AX=XB on the euclidean group.
  ///        Robotics and Automation, IEEE Transactions on, 10(5):717--721;1994.
  ///
  /// \param A The stack of relative frame transforms for sensor 1 
  ///          (e.g., wrist relative to istelf after arbitrary movement)
  ///          size: 4n x 4, for n measurements
  /// \param B The corresponding stack of relative frame transforms for sensor 2
  ///          (e.g., sensor frame relative to itself after movement)
  ///          size: 4n x 4, for n measurements
  /// \param X The resuling relationship between sensor 1 and sensor 2.
  void ax_xb(const Eigen::MatrixX4d &A, const Eigen::MatrixX4d &B, Eigen::Matrix4d &X);

  /// Do principal component analysis on a matrix
  /// \param X The matrix
  /// \return The principal component basis
  Eigen::MatrixXd princomp(const Eigen::MatrixXd &X);

  /// Fit a 3D line to a set of points
  ///
  /// This uses principal component analysis to fit a line to the data
  ///
  /// \param[in] points An n*3 matrix of points
  /// \param[out] point A point on the fit line
  /// \param[out] vec The direction of the line
  void fit3DLine(const Eigen::MatrixXd &points, Eigen::Vector3d &point, Eigen::Vector3d &vec);

  /// Solve for a 3D transformation with known correspondences
  /// \param ptsMoving Set of moving points
  /// \param ptsFixed Set of fixed points
  /// \param T Transform such that ptsFixed = T * ptsMoving
  /// \return true on valid transform
  /// \todo consider replacing with Rob's transform function.
  bool computeTransform(const Eigen::Matrix3Xd &ptsMoving, const Eigen::Matrix3Xd &ptsFixed, Eigen::Affine3d &T);

  /// Solve for a 3D transformation with unknown correspondences
  /// \param ptsMoving Set of moving points. This is updated on completion to reflect the correct ordering
  /// \param ptsFixed Set of fixed points
  /// \param T Transform such that ptsFixed = T * ptsMoving
  /// \param ordering Updated with the ordering of ptsMoving such that for i = old location, ordering(i) = new location.
  /// \return true on valid transform
  /// \note This is valid for up to a 5-element vector
  bool computeCorrespondencelessTransform(Eigen::Matrix3Xd &ptsMoving,
      const Eigen::Matrix3Xd &ptsFixed, Eigen::Affine3d &T, Eigen::VectorXi &ordering
      /*= Eigen::VectorXi()*/);

}

#endif // _BIGSS_MATH_H

