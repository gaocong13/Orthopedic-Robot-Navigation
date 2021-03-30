#include "bigssMath.h"

#include <iostream>
#include <algorithm>

#include <cisstVector/vctFixedSizeMatrixTypes.h>

#if USE_CISST
#include <cisstNumerical/nmrSVD.h>
#include <cisstNumerical/nmrPInverse.h>
#endif

#include <Eigen/Dense>

/// Convert from a cisst dynamic matrix to an eigen representation
/// \param X The cisst representation
/// \return The eigen representation
Eigen::MatrixXd BIGSS::convertCisstToEigen(const vctDynamicMatrix<double> &X)
{
  Eigen::MatrixXd xeig = Eigen::MatrixXd::Zero(X.rows(), X.cols());

  for (unsigned int i = 0; i < X.rows(); i++) {
    for (unsigned int j = 0; j < X.cols(); j++) {
      xeig(i, j) = X(i, j);
    }
  }
  return xeig;
}

/// Convert from an eigen matrix to cisst representation
/// \param X the eigen representation
/// \return The cisst representation
vctDynamicMatrix<double> BIGSS::convertEigenToCisst(const Eigen::MatrixXd &X)
{
  vctDynamicMatrix<double> xcisst(X.rows(), X.cols());
  for (auto i = 0; i < X.rows(); i++) {
    for (auto j = 0; j < X.cols(); j++) {
      xcisst(i, j) = X(i, j);
    }
  }
  return xcisst;
}

void BIGSS::ax_xb(vctDoubleMat &A, vctDoubleMat &B, vctFrm4x4 &X)
{

#if USE_CISST
  vctDouble3x3 mList, xtmp, mat;
  vctRot3 rotMat;
  vctDynamicMatrixRef<double> Atmp, Btmp;
  vctAxAnRot3 Arot, Brot;
  size_t nX = A.rows() / 4;

  mat.SetAll(0.0);
  mList.SetAll(0.0);
  xtmp.SetAll(0.0);

  vctDynamicMatrix<double> C(3 * nX, 3);
  vctDynamicVector<double> d(3 * nX);
  vctDynamicMatrixRef<double> Cref;
  vctDynamicVectorRef<double> dref;

  const unsigned int size = 3;
  vctDynamicMatrix<double> U(size, size);
  vctDynamicMatrix<double> Vt(size, size);
  vctDynamicVector<double> S(size);
  vctDynamicMatrix<double> svdCopy;

  for (size_t i = 0; i < nX; i++) {
    Atmp.SetRef(A, 4 * i, 0, 3, 3);
    Btmp.SetRef(B, 4 * i, 0, 3, 3);

    rotMat.Assign(Atmp);
    Arot.From(rotMat);
    rotMat.Assign(Btmp);
    Brot.From(rotMat);

    xtmp.OuterProductOf(Brot.Axis(), Arot.Axis());

    mList += Brot.Angle() * Arot.Angle() * xtmp;
  }

  svdCopy = mList;
  try {
    nmrSVD(svdCopy, U, S, Vt);
  }
  catch (...) {
    std::cout << "An exception occured, check cisstLog.txt." << std::endl;
  }

  mat.Diagonal().Assign(1 / S.X(), 1 / S.Y(), 1 / S.Z());
  vctFixedSizeMatrix<double, 3, 3> vv = Vt;
  xtmp = vv.Transpose() * mat * vv * mList.Transpose();
  X.Rotation().FromNormalized(vctRot3(xtmp));

  for (size_t i = 0; i < nX; i++) {
    Cref.SetRef(C, 3 * i, 0, 3, 3);
    dref.SetRef(d, 3 * i, 3);

    Atmp.SetRef(A, 4 * i, 0, 3, 3);
    Cref = vctMat::Eye(3) - Atmp;
    Atmp.SetRef(A, 4 * i, 3, 3, 1);
    Btmp.SetRef(B, 4 * i, 3, 3, 1);
    vctDynamicMatrix<double> cc = Atmp - vctDynamicMatrix<double>(xtmp) * Btmp;
    dref = cc.Column(0);
  }
  vctDynamicMatrix<double> inv(3, 3);
  vctDynamicMatrix<double> P = C.Transpose() * C;
  nmrPInverse(P, inv);
  X.Translation().Assign(inv * C.Transpose() * d);
#else
  size_t k = A.rows();
  Eigen::MatrixXd A_eig = Eigen::MatrixXd::Zero(k, 4);
  Eigen::MatrixXd B_eig = Eigen::MatrixXd::Zero(k, 4);
  Eigen::Matrix4d X_eig;

  for (size_t i = 0; i < k; i++)
  {
    A_eig.row(i) << A.Row(i).X(), A.Row(i).Y(), A.Row(i).Z(), A.Row(i).W();
    B_eig.row(i) << B.Row(i).X(), B.Row(i).Y(), B.Row(i).Z(), B.Row(i).W();
  }
  BIGSS::ax_xb(A_eig, B_eig, X_eig);

  X.Rotation().FromNormalized(vctRot3(vct3x3(
    X_eig(0, 0), X_eig(0, 1), X_eig(0, 2),
    X_eig(1, 0), X_eig(1, 1), X_eig(1, 2),
    X_eig(2, 0), X_eig(2, 1), X_eig(2, 2)
  )));
  X.Translation().Assign(X_eig(0, 3), X_eig(1, 3), X_eig(2, 3));
#endif // USE_CISST
}

void BIGSS::circleFitToOrigin(std::vector< vctFrm4x4 > data, vctFrm4x4 &centerPose, double &radius)
{
  vctDynamicMatrix<double> p(data.size(), 3);
  vctDynamicMatrix<double> pCentered = p;
  vct3 pMean(0.0, 0.0, 0.0);

  for(unsigned int i=0; i<data.size(); i++) {
    p.Row(i).Assign( data[i].Translation() );
    pMean += vct3(p.Row(i));
  }

  pMean /= static_cast<double>(data.size());

  for(unsigned int i=0; i<data.size(); i++) {
    pCentered.Row(i).Assign( vct3(p.Row(i)) - pMean );
  }

  // fit a plane
  // AB = 0, B = [b1 b2 b3 b4]
  // b1*X + b2*Y + b3*Z = B4 = 0
  vctDynamicMatrix<double> A(data.size(), 4, 1.0, VCT_COL_MAJOR);
  vctDynamicMatrixRef<double> ARef;
  ARef.SetRef(A, 0, 0, data.size(), 3);
  ARef.Assign(pCentered);

#if CISST_HAS_NETLIB
  vctDynamicMatrix<double> Vt(4, 4, VCT_COL_MAJOR);
  vctDynamicMatrix<double> input = A;
  vctDynamicMatrix<double> U(data.size(), 4, VCT_COL_MAJOR);
  vctDynamicVector<double> S(4);
  try {
    nmrSVDEconomy(input, U, S, Vt);
  } catch( ... ) {
    vtkErrorMacro(<< "Unable to compute SVD, check cisstLog.txt");
    centerPose = vctFrm4x4();
    radius = 0;
    return;
  }

  // Grab row since this is the transpose.
  vct3 Z(Vt.Row(3).X(), Vt.Row(3).Y(), Vt.Row(3).Z());
#else

  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(data.size(), 4);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Ad, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXd v = svd.matrixV();
  
  vct3 Z(v.row(3).x(), v.row(3).y(), v.row(3).z());
#endif

  vct3 X(0.0, 0.0, -1.0);
  vct3 Y = vctCrossProduct(Z, X);
  X = vctCrossProduct(Y, Z);

  vctDynamicMatrix<double> coeff(3, 3);
  coeff.Column(0).Assign(X);
  coeff.Column(1).Assign(Y);
  coeff.Column(2).Assign(Z);

  vctDynamicMatrix<double> fit = pCentered * coeff;

  // fit a circle to XY plane
  vctDynamicVector<double> XPts = fit.Column(0);
  vctDynamicVector<double> YPts = fit.Column(1);
  vctDynamicVector<double> X2 = XPts;
  vctDynamicVector<double> Y2 = YPts;
  vctDynamicVector<double> XY;
  vctDynamicMatrix<double> CircleMat(data.size(), 3, 1.0);
  CircleMat.Column(0) = XPts;
  CircleMat.Column(1) = YPts;

  vctDynamicMatrix<double> CircleMatInv(3, data.size());

  X2.ElementwiseProductOf(XPts, XPts);
  Y2.ElementwiseProductOf(YPts, YPts);
  XY = X2 + Y2;
  XY *= -1;

#if CISST_HAS_NETLIB
  nmrPInverse(CircleMat, CircleMatInv);
#else
  Eigen::MatrixXd ceig = Eigen::MatrixXd::Zero(data.size(), 3);

  for (unsigned int i = 0; i < data.size(); i++)
  {
    ceig(i, 0) = CircleMat(i, 0);
    ceig(i, 1) = CircleMat(i, 1);
    ceig(i, 2) = CircleMat(i, 2);
  }

  // todo
  // perform pseudoinverse
  Eigen::MatrixXd cinv;

  for (unsigned int i = 0; i < data.size(); i++)
  {
    CircleMatInv(i, 0) = cinv(i, 0);
    CircleMatInv(i, 1) = cinv(i, 1);
    CircleMatInv(i, 2) = cinv(i, 2);
  }
#endif

  vct3 a;
  a = CircleMatInv * XY;

  double xc = -0.5 * a[0];
  double yc = -0.5 * a[1];
  radius = sqrt( (a[0]*a[0] + a[1]*a[1])/4 - a[2] );
  centerPose.Translation().Assign( coeff * vctDynamicVector<double>(3, xc, yc, 0.0) + vctDynamicVector<double>(pMean) );
}

bool BIGSS::rotationCalibration(std::vector< vctFrm4x4 > data, vctFrm4x4 &mat)
{
  vctFrm4x4 centerPose;

  auto mRows = data.size()-1;
  int mCols = 3;
  unsigned int i = 0;

  if( mRows < 4) {
    // want at least 4 frames
    return false;
  }

  // compute the center axis
  vctFrm4x4 base = data[0];
  vctDynamicMatrix<double> relRodAng(mRows, mCols, VCT_COL_MAJOR);
  vct3 centroid(0.0, 0.0, 0.0);

  for(i=1; i < data.size(); i++) {
    vctRodriguezRotation3<double> rodAng( vctRot3( base.Rotation().Transpose() * data[i].Rotation() ).Normalized() );
    relRodAng.Row(i-1).Assign(rodAng);
    centroid += rodAng;
  }

  centroid /= (static_cast<double>(data.size())-1.0);

  // Line Fitting in 3D
#ifdef CISST_HAS_NETLIB

  // translate the points to the centroid.
  vctDynamicMatrix<double> linePts = relRodAng;
  for(i=0; i < mRows; i++) {
    linePts.Row(i).X() -= centroid.X();
    linePts.Row(i).Y() -= centroid.Y();
    linePts.Row(i).Z() -= centroid.Z();
  }

  vctDynamicMatrix<double> Vt(mCols, mCols, VCT_COL_MAJOR);
  vctDynamicVector<double> S(mCols);

  vctDynamicMatrix<double> U(mRows, mCols, VCT_COL_MAJOR);

  try {
    nmrSVDEconomy(linePts, U, S, Vt);
  } catch( ... ) {
    vtkErrorMacro( << "Unable to compute SVD for rotation calibration");
    return false;
  }

  // find the index of the largest singular value and the corresponding
  // right singular vector
  double max = -1 * std::numeric_limits<double>::max();
  int maxidx = 0;
  for (i = 0; i<S.size(); i++) {
    if (max < S[i]) {
      max = S[i];
      maxidx = i;
    }
  }

  // Grab row since this is the transpose.
  vct3 Z = Vt.Row(maxidx);

#else

  // translate the points to the centroid.
  Eigen::MatrixXd linePts = Eigen::MatrixXd::Zero(relRodAng.rows(), relRodAng.cols());
  for (i = 0; i < mRows; i++) {
    linePts(i, 0) = relRodAng.Row(i).X() - centroid.X();
    linePts(i, 1) = relRodAng.Row(i).Y() - centroid.Y();
    linePts(i, 2) = relRodAng.Row(i).Z() - centroid.Z();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(linePts, Eigen::ComputeThinU | Eigen::ComputeThinV);
  
  Eigen::VectorXd sv = svd.singularValues();
  Eigen::VectorXd::Index max;
  sv.maxCoeff(&max);

  Eigen::MatrixXd v = svd.matrixV();

  vct3 Z(v.row(max).x(), v.row(max).y(), v.row(max).z());
#endif

  // End line fitting

  vctDynamicVector<double> angles(data.size());
  vctDynamicVectorRef<double> anglesSubRef(angles, 1, data.size()-1);
  anglesSubRef.Assign( relRodAng * vctDynamicVector<double>(Z) );
  angles[0] = 0;

  // compute center of frames by fitting a circle
  double radius;
  BIGSS::circleFitToOrigin(data, centerPose, radius);

  vct3 X = base.Rotation().Transpose() * (data[0].Translation()-centerPose.Translation()).Normalized();
  vct3 Y = vctCrossProduct(Z, X);
  Y.NormalizedSelf();
  X = vctCrossProduct(Y, Z);
  X.NormalizedSelf();
  vct3x3 coeff;
  coeff.Column(0).Assign(X);
  coeff.Column(1).Assign(Y);
  coeff.Column(2).Assign(Z);
  centerPose.Rotation().Assign( base.Rotation() * coeff );

  // compute translation and rotation part of offset
  vctDynamicMatrix<double> C(data.size()*3, 3, 0.0, VCT_COL_MAJOR);
  vctDynamicMatrix<double> CInv(3, data.size() * 3, VCT_COL_MAJOR);
  vctDynamicVector<double> d(data.size()*3, 0.0);
  vct3 OffsetRod(0.0, 0.0, 0.0);
  vctDynamicMatrixRef<double> CSubRef;
  vctDynamicVectorRef<double> dSubRef;
  vct3 ZAxis(0.0, 0.0, 1.0);

  for(i=0; i < data.size(); i++) {
    CSubRef.SetRef(C, i*3, 0, 3, 3);
    dSubRef.SetRef(d, i*3, 3);

    vctMatrixRotation3<double> zRot(vctAxisAngleRotation3<double> (ZAxis, angles[i]));
    CSubRef.Assign( centerPose.Rotation() * zRot );
    dSubRef.Assign( data[i].Translation() - centerPose.Translation() );

    vctRodriguezRotation3<double> rodAng( vctRot3( (centerPose.Rotation() * zRot).Transpose() * data[i].Rotation() ).Normalized() );
    OffsetRod += rodAng;
  }

#if CISST_HAS_NETLIB
  nmrPInverse(C, CInv);
#else
  Eigen::MatrixXd ceig = Eigen::MatrixXd::Zero(data.size() * 3, 3);

  for (unsigned int i = 0; i < data.size() * 3; i++)
  {
    ceig(i, 0) = C(i, 0);
    ceig(i, 1) = C(i, 1);
    ceig(i, 2) = C(i, 2);
  }

  // todo
  // perform pseudoinverse
  Eigen::MatrixXd cinv;

  for (unsigned int i = 0; i < data.size() * 3; i++)
  {
    CInv(i, 0) = cinv(i, 0);
    CInv(i, 1) = cinv(i, 1);
    CInv(i, 2) = cinv(i, 2);
  }
  
#endif

  mat.Translation().Assign( CInv*d );

  OffsetRod /= static_cast<double>(data.size());
  mat.Rotation().Assign( vctMatrixRotation3<double>( vctRodriguezRotation3<double>(OffsetRod) ) );

  return true;
}

bool BIGSS::pivotCalibration( std::vector< vctFrm4x4 > frames, vct3 &tooltip, vct3 &pivot, double &errorRMS )
{

  // Create equations
  auto numPoints = frames.size();

  if( numPoints < 4 ) {
    // want at least 4 frames
    std::cerr << "PivotCalibration: too few frames" << std::endl;
    return false;
  }


#if CISST_HAS_NETLIB
  // compute pivot calibration from a series of measured frames (vctFrm4x4)
  // ( method is based on Prof. Taylor's CIS class )
  // this is pulled from mtsNDISerial.cpp

  vctMat A(3 * numPoints, 6, VCT_COL_MAJOR);
  vctMat b(3 * numPoints, 1, VCT_COL_MAJOR);

  for (unsigned int i = 0; i < numPoints; i++) {

    vctDynamicMatrixRef<double> rotation(3, 3, 1, numPoints*3, A.Pointer(i*3, 0));
    rotation.Assign(frames[i].Rotation());

    vctDynamicMatrixRef<double> identity(3, 3, 1, numPoints*3, A.Pointer(i*3, 3));
    identity.Assign(-vctRot3::Identity());

    vctDynamicVectorRef<double> translation(3, b.Pointer(i*3, 0));
    translation.Assign(frames[i].Translation());
  }

  nmrLSSolver calibration(A, b);
  calibration.Solve(A, b);

  for (unsigned int i = 0; i < 3; i++) {
    tooltip.Element(i) = -b.at(i, 0);
    pivot.Element(i) = -b.at(i+3, 0);
  }
#else
  Eigen::MatrixXd A = Eigen::MatrixXd(3 * numPoints, 6);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(3 * numPoints);

  for (unsigned int i = 0; i < numPoints; i++) {

    A.block<1, 3>(3 * i, 0) << frames[i].Rotation().Row(0).X(), frames[i].Rotation().Row(0).Y(), frames[i].Rotation().Row(0).Z();
    A.block<1, 3>(3 * i + 1, 0) << frames[i].Rotation().Row(1).X(), frames[i].Rotation().Row(1).Y(), frames[i].Rotation().Row(1).Z();
    A.block<1, 3>(3 * i + 2, 0) << frames[i].Rotation().Row(2).X(), frames[i].Rotation().Row(2).Y(), frames[i].Rotation().Row(2).Z();
    A.block<3, 3>(3 * i, 3) = -Eigen::Matrix3d::Identity();
    b.segment<3>(3 * i) << frames[i].Translation().X(), frames[i].Translation().Y(), frames[i].Translation().Z();
  }

  Eigen::VectorXd soln = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);


  for (unsigned int i = 0; i < 3; i++) {
    tooltip.Element(i) = -soln(i);
    pivot.Element(i) = -soln(i + 3);
  }
#endif

  vct3 error;
  double errorSquareSum = 0.0;
  for (unsigned int i = 0; i < numPoints; i++) {
    error = (frames[i] * tooltip) - pivot;
//    CMN_LOG_CLASS_RUN_DEBUG << error << std::endl;
    errorSquareSum += error.NormSquare();
  }
  errorRMS = sqrt(errorSquareSum / numPoints);

  return true;
}

vctDynamicMatrix<double> BIGSS::princomp(const vctDynamicMatrix<double> &X)
{
  Eigen::MatrixXd xeig = convertCisstToEigen(X);
  Eigen::MatrixXd Vt = princomp(xeig);
  return convertEigenToCisst(Vt);
}

void BIGSS::fit3DLine(const vctDynamicMatrix<double> &points, vct3 &point, vct3 &vec)
{
  Eigen::MatrixXd pts = convertCisstToEigen(points);
  Eigen::Vector3d p, v;
  fit3DLine(pts, p, v);
  point.Assign(p[0], p[1], p[2]);
  vec.Assign(v[0], v[1], v[2]);
}


