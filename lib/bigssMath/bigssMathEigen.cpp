#include "bigssMathEigen.h"

#include <iostream>
#include <algorithm>

#include <Eigen/Dense>

void BIGSS::ax_xb(const Eigen::MatrixX4d &A, const Eigen::MatrixX4d &B, Eigen::Matrix4d &X)
{
  Eigen::Matrix3d mList, xtmp, mat;
  Eigen::Vector3d rotMat;

  Eigen::MatrixXd C;
  Eigen::VectorXd d;

  size_t nX = A.rows() / 4;

  mat.setZero();
  mList.setZero();
  xtmp.setZero();

  X.setIdentity();

  C = Eigen::MatrixXd::Zero(3 * nX, 3);
  d = Eigen::VectorXd::Zero(3 * nX);

  for (size_t i = 0; i<nX; i++) {
    Eigen::Matrix3d ablk = A.block(4 * i, 0, 3, 3);
    Eigen::Matrix3d bblk = B.block(4 * i, 0, 3, 3);
    std::cout << ablk << std::endl << std::endl;
    Eigen::AngleAxisd arot, brot;
    arot.fromRotationMatrix(ablk);
    brot.fromRotationMatrix(bblk);

    Eigen::Vector3d aax = arot.axis();
    Eigen::Vector3d bax = brot.axis();

    xtmp = bax * aax.transpose();

    mList += brot.angle() * arot.angle() * xtmp;
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(mList, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d sv = svd.singularValues();
  Eigen::Matrix3d v = svd.matrixV();

  mat(0, 0) = 1 / sv[0];
  mat(1, 1) = 1 / sv[1];
  mat(2, 2) = 1 / sv[2];

  xtmp = v * mat * v.transpose() * mList.transpose();
  std::cout << xtmp << std::endl;

  Eigen::AngleAxisd xax;
  xax.fromRotationMatrix(xtmp);

  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  for (size_t i = 0; i<nX; i++) {
    Eigen::MatrixXd dref = d.segment(3 * i, 3);

    Eigen::MatrixXd ablk = A.block(4 * i, 0, 3, 3);
    C.block(3 * i, 0, 3, 3) = I - ablk;
    Eigen::VectorXd aref = A.block(4 * i, 3, 3, 1);
    Eigen::VectorXd bblk = B.block(4 * i, 3, 3, 1);
    Eigen::VectorXd cc = aref - xtmp * bblk;
    d.segment(3 * i, 3) = cc;
  }

  Eigen::MatrixXd P = C.transpose() * C;
  Eigen::Vector3d trans = P.inverse() * C.transpose() * d;

  X.block<3, 3>(0, 0) = xtmp;
  X.block<3, 1>(0, 3) = trans;
}

Eigen::MatrixXd BIGSS::princomp(const Eigen::MatrixXd &X)
{
  Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinV);
  Eigen::MatrixXd Vt = svd.matrixV();

  return Vt;
}

void BIGSS::fit3DLine(const Eigen::MatrixXd &points, Eigen::Vector3d &point, Eigen::Vector3d &vec)
{
  Eigen::MatrixXd W = princomp(points);
  vec = W.col(0);
  point = points.colwise().mean();
}

bool BIGSS::computeTransform(const Eigen::Matrix3Xd &ptsMoving, const Eigen::Matrix3Xd &ptsFixed, Eigen::Affine3d &T)
{
  T.setIdentity();

  if (ptsMoving.cols() != ptsFixed.cols())
    return false;

  if (ptsMoving.cols() < 3)
    return false;

  Eigen::Matrix3Xd aBar = ptsMoving.colwise() - ptsMoving.rowwise().mean();
  Eigen::Matrix3Xd bBar = ptsFixed.colwise() - ptsFixed.rowwise().mean();

  Eigen::Matrix3d H = aBar * bBar.transpose();

  double traceH = H.trace();

  Eigen::Vector3d delta;
  delta(0) = H(1, 2) - H(2, 1);
  delta(1) = H(2, 0) - H(0, 2);
  delta(2) = H(0, 1) - H(1, 0);

  Eigen::Matrix4d G;
  G(0, 0) = traceH;
  G.block<1, 3>(0, 1) = delta.transpose();
  G.block<3, 1>(1, 0) = delta;
  G.block<3, 3>(1, 1) = H + H.transpose() - (traceH * Eigen::Matrix3d::Identity());

  Eigen::EigenSolver<Eigen::Matrix4d> eig(G, true);
  Eigen::Matrix4cd evecs = eig.eigenvectors();
  Eigen::Vector4cd evals = eig.eigenvalues();
  Eigen::Vector4d::Index idx;
  //evals.maxCoeff(&idx);
  evals.real().maxCoeff(&idx);
  Eigen::Vector4cd ee = evecs.col(idx);
  //Eigen::Vector4d evec = ee.cast<Eigen::Vector4d>();
  //Eigen::Vector4d evec = evecs.col(idx).cast<Eigen::Vector4d>();

  Eigen::Quaterniond quat(ee(0).real(), ee(1).real(), ee(2).real(), ee(3).real());
  T = T.rotate(quat);
  Eigen::Vector3d p = ptsFixed.rowwise().mean() - T * ptsMoving.rowwise().mean();
  T = T.pretranslate(p);

  return true;
}

bool BIGSS::computeCorrespondencelessTransform(Eigen::Matrix3Xd &ptsMoving, const Eigen::Matrix3Xd &ptsFixed, Eigen::Affine3d &T, Eigen::VectorXi &ordering)
{
  T.setIdentity();
  Eigen::Affine3d guessT;
  int nPts = ptsMoving.cols();

  if (nPts != ptsFixed.cols())
    return false;

  if (nPts > 5)
    return false;

  // compute number of permutations
  int nPerms = 1;
  for (int i = 1; i < nPts; i++)
    nPerms *= (i + 1);

  // generate the permutations
  Eigen::VectorXi indices;
  indices.setLinSpaced(nPts, 0, nPts - 1);

  Eigen::Matrix3Xd bestPts;

  double maxError = std::numeric_limits<double>::max();
  Eigen::VectorXd errors;
  errors.setZero(nPerms);
  int i = 0;
  do// (int i = 0; i < nPerms; i++)
  {
    Eigen::Matrix3Xd newPts = ptsMoving;

    // find the next permutation
    //std::next_permutation(indices.data(), indices.data() + nPts);
    for (int j = 0; j < nPts; j++)
    {
      newPts.col(indices(j)) = ptsMoving.col(j);
    }

    // compute the transform
    computeTransform(newPts, ptsFixed, guessT);

    // compute the error
    Eigen::MatrixXd dt = guessT * newPts - ptsFixed;
    Eigen::VectorXd res = dt.colwise().norm();
    errors(i) = res.sum();
    if (errors(i) < maxError)
    {
      ordering = indices;
      maxError = errors(i);
      bestPts = newPts;
      T = guessT;
    }
    i++;
  } while (std::next_permutation(indices.data(), indices.data() + nPts));

  ptsMoving = bestPts;

  return true;
}

