/// Test file for solving AX = XB

#include "bigssMath.h"

#include <Eigen/SVD>

#include <cisstVector/vctTransformationTypes.h>
#include <cisstVector/vctFixedSizeVectorTypes.h>
#include <cisstVector/vctRandom.h>
#include <cisstCommon/cmnRandomSequence.h>

/// conduct a sensitivity analysis of AX=XB problem
/// using simulation data
/// input     X: nominal transformation (answer)
///           max_init_trans: maximum transformation for initial A matrix
///           k: number of sample pairs which we use for the simulation
///           rot_noise: maximum rotational noise which we add to the data
///           trans_noise: maximum translation noise which we add to the data
/// output    err_trans: translational error between X and estimated X
///           err_rot: rotational error between X and estimated X (length of
///                    the rodriguez angle vector in the unit of "degree")

void AX_XB_Simulation_OneTry(vctFrm4x4 &X, /*double max_init_trans,*/ int k, /*double rot_noise, double trans_noise,*/ vctFrm4x4 &Xest)
{

  // generate uncorrupted pairs
  vctDynamicMatrix<double> A_true(4*k, 4);
  vctDynamicMatrix<double> B_true(4*k, 4);

  vctDynamicMatrixRef<double> Aref, Bref;
  A_true.SetAll(0.0);
  B_true.SetAll(0.0);

  vctFrm4x4 Xinv = X.Inverse();

  // generate random rodriguez angle
  // double rot_step = 2*cmnPI/(k+1);
  vctFrm4x4 rot_base;
  rot_base.Rotation().From(vctRot3(1, 0, 0, 0, 0, -1, 0, 1, 0));
  rot_base.Translation().Assign(0.0, 0.0, 550.0);

  vct3 ax;
  double ang;
  vctFrm4x4 frm;

  vctDynamicMatrix<double> XDinv, XD;
  XDinv = Xinv;
  XD = X;
  cmnRandomSequence & randomSequence = cmnRandomSequence::GetInstance();
  for(int i=0; i<k; i++) {
    ang = randomSequence.ExtractRandomDouble(0, 2*cmnPI);
    vctRandom(ax, -1.0, 1.0);
    ax.NormalizedSelf();
    frm.Rotation().From(vctAxAnRot3(ax, ang));
    Aref.SetRef(A_true, 4*i, 0, 4, 4);
    Bref.SetRef(B_true, 4*i, 0, 4, 4);
    Aref.Assign(rot_base * frm * rot_base.Inverse());
    Bref.Assign(XDinv*Aref*XD);
  }

  std::cout << A_true << std::endl << std::endl;
  std::cout << B_true << std::endl << std::endl;

  // solve AX=XB problem from noise-added data
  BIGSS::ax_xb(A_true, B_true, Xest);
//#else
//  // convert cisst to eigen
//  Eigen::MatrixXd A_eig = Eigen::MatrixXd::Zero(4 * k, 4);
//  Eigen::MatrixXd B_eig = Eigen::MatrixXd::Zero(4 * k, 4);
//  Eigen::Matrix4d X_eig;
//
//  for (int i = 0; i < 4*k; i++)
//  {
//    A_eig.row(i) << A_true.Row(i).X(), A_true.Row(i).Y(), A_true.Row(i).Z(), A_true.Row(i).W();
//    B_eig.row(i) << B_true.Row(i).X(), B_true.Row(i).Y(), B_true.Row(i).Z(), B_true.Row(i).W();
//  }
//  BIGSS::ax_xb(A_eig, B_eig, X_eig);
//
//  Xest.Rotation().FromNormalized(vctRot3(vct3x3(
//    X_eig(0, 0), X_eig(0, 1), X_eig(0, 2),
//    X_eig(1, 0), X_eig(1, 1), X_eig(1, 2),
//    X_eig(2, 0), X_eig(2, 1), X_eig(2, 2)
//  )));
//  Xest.Translation().Assign(X_eig(0, 3), X_eig(1, 3), X_eig(2, 3));
//#endif
  //Xest = AX_XB( A_noised, B_noised );
  //  err_trans = norm(X(1:3,4)-X_estimated(1:3,4));
  //err_rot = (180/pi)*norm(RotMat2RodAng(X(1:3,1:3)'*X_estimated(1:3,1:3)));
}

int main()
{
  vct3 rot_axis(0.5, 0.5, 0.2);
  //double angle = 40;
  int k=5;

  vctFrm4x4 X, Xest;
  X.Rotation().FromNormalized(vctRot3(vct3x3(-0.502039,    -0.819662,    -0.275883, 
				 0.817477,    -0.345628,    -0.460731, 
				 0.282291,    -0.456832,     0.843573)));
  X.Translation().Assign(72.9024, -147.249,  -95.0182);

  //double num_trial = 100; //< number of trial for each condition
  //double rot_noise = cmnPI/100; //< maximum noise added to the rotation component
  //double trans_noise = 3; //< maximum noise added to the translation component

  AX_XB_Simulation_OneTry(X, /*100*sqrt(3.0),*/ k, /*rot_noise, trans_noise,*/ Xest);

  std::cout << X << std::endl << std::endl;
  std::cout << Xest << std::endl << std::endl;

  // test pseudoinverse
  Eigen::MatrixXd jp_eig = Eigen::MatrixXd::Random(6,6);
  Eigen::JacobiSVD<Eigen::MatrixXd> svdOfM(jp_eig, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::MatrixXd U = svdOfM.matrixU();
  const Eigen::MatrixXd V = svdOfM.matrixV();
  const Eigen::MatrixXd S = svdOfM.singularValues();
  double tolerance = 1.0e-6;

  Eigen::MatrixXd Sinv = S;
  double maxsv = 0;
  for (unsigned int i = 0; i < S.rows(); ++i)
    if (fabs(S(i)) > maxsv) maxsv = fabs(S(i));
  for (unsigned int i = 0; i < S.rows(); ++i)
  {
    //Those singular values smaller than a percentage of the maximum singular value are removed
    if (fabs(S(i)) > maxsv * tolerance)
      Sinv(i) = 1.0 / S(i);
    else Sinv(i) = 0;
  }
  
  Eigen::MatrixXd jpinv_eig = V * Sinv.asDiagonal() * U.transpose();
  Eigen::MatrixXd jpinv_comp;

  Eigen::MatrixXd::pinv(jp_eig, jpinv_comp, 1.0e-6);

  std::cout << std::endl << jpinv_comp << std::endl << std::endl;;
  std::cout << jpinv_eig << std::endl;
}
