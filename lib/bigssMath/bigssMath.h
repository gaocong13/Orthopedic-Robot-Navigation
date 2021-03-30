#ifndef _BIGSS_MATH_H
#define _BIGSS_MATH_H

// If CISST netlib is available and CISST vectors are used, then we use
// CISST netlib to solve the problem. Otherwise, we default to Eigen.
#define USE_CISST CISST_HAS_NETLIB

#include <cisstVector/vctDynamicMatrixTypes.h>
#include <cisstVector/vctTransformationTypes.h>

#include <bigssMathEigen.h>

namespace BIGSS {
  /// Convert from a cisst dynamic matrix to an eigen representation
  /// \param X The cisst representation
  /// \return The eigen representation
  Eigen::MatrixXd convertCisstToEigen(const vctDynamicMatrix<double> &X);

  /// Convert from an eigen matrix to cisst representation
  /// \param X the eigen representation
  /// \return The cisst representation
  vctDynamicMatrix<double> convertEigenToCisst(const Eigen::MatrixXd &X);

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
  void ax_xb(vctDoubleMat &A, vctDoubleMat &B, vctFrm4x4 &X);

  /// Perform a circle fit to the origin from a set of frames
  /// \param data A vector of 4x4 frames
  /// \param centerPose The fit center pose
  /// \param radius The radius of the circle
  void circleFitToOrigin(std::vector< vctFrm4x4 > data, vctFrm4x4 &centerPose, double &radius);

  /// Perform a rotation calibration
  /// \param data A vector of 4x4 frames
  /// \param mat The calibrated rotation matrix
  /// \return True on success
  bool rotationCalibration(std::vector< vctFrm4x4 > data, vctFrm4x4 &mat);

  /// Perform a pivot calibration
  /// \param frames Vector of 4x4 frames
  /// \param tooltip The computed tool tip
  /// \param pivot The computed pivot point
  /// \param errorRMS The RMS error during the calibration
  /// \return True on success
  bool pivotCalibration(std::vector< vctFrm4x4 > frames, vct3 &tooltip, vct3 &pivot, double &errorRMS);

  /// Do principal component analysis on a matrix
  /// \param X The matrix
  /// \return The principal component basis
  vctDynamicMatrix<double> princomp(const vctDynamicMatrix<double> &X);

  /// Fit a 3D line to a set of points
  ///
  /// This uses principal component analysis to fit a line to the data
  ///
  /// \param[in] points An n*3 matrix of points
  /// \param[out] point A point on the fit line
  /// \param[out] vec The direction of the line
  void fit3DLine(const vctDynamicMatrix<double> &points, vct3 &P, vct3 &vec);

}

#endif // _BIGSS_MATH_H

