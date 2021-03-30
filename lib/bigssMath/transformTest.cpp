/// Test file for solving a transformation

#include "bigssMath.h"
#include <cmath>

#include <Eigen/SVD>

// #define M_PI 3.14159

int main()
{
  int nPts = 4;
  Eigen::Matrix3Xd moving = 100 * Eigen::Matrix3Xd::Random(3,nPts);
  Eigen::Affine3d T, T2;
  T.setIdentity();
  T.translate(Eigen::Vector3d(10,20,30));
  T.rotate(Eigen::AngleAxisd(0.5*M_PI, Eigen::Vector3d::UnitY()));

  T2.setIdentity();

  Eigen::Matrix3Xd fixed = T * moving;

  BIGSS::computeTransform(moving, fixed, T2);

  Eigen::Matrix4d diff =  T.matrix() - T2.matrix();
  std::cout << diff.norm() << std::endl;

  std::cout << T2 * moving - fixed << std::endl << std::endl;

  Eigen::Matrix3Xd cmoving = moving.leftCols(nPts);
  Eigen::Matrix3Xd cfixed = fixed.leftCols(nPts);
  Eigen::Affine3d T3;
  Eigen::VectorXi ordering;

  cmoving.col(0) = moving.col(1);
  cmoving.col(1) = moving.col(3);
  cmoving.col(2) = moving.col(0);
  cmoving.col(3) = moving.col(2);


  std::cout << moving << std::endl << std::endl;
  std::cout << cmoving << std::endl << std::endl;
  BIGSS::computeCorrespondencelessTransform(cmoving, cfixed, T3, ordering);

  std::cout << "Transform (truth): " << std::endl << T.matrix() << std::endl << std::endl;
  std::cout << "Transform (with correspondence): " << std::endl << T2.matrix() << std::endl << std::endl;
  std::cout << "Transform (correspondenceless): " << std::endl << T3.matrix() << std::endl << std::endl;
  std::cout << ordering << std::endl << std::endl;
  std::cout << "original: " << std::endl << moving << std::endl << std::endl;
  std::cout << "permuted: " << std::endl << cmoving << std::endl << std::endl;
  std::cout << "Press any key to exit..." << std::endl;
  std::getchar();
}
