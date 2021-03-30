#include "smoothingSpline.h"

#include <Eigen/Geometry>

using namespace BIGSS;

smoothingSpline::smoothingSpline(const Eigen::VectorXd &y, const double lambda)
{
  Eigen::VectorXd x;
  x.setLinSpaced(y.size(), 0, 1);
  createSpline(x, y, lambda);
}

smoothingSpline::smoothingSpline(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double lambda)
{
  createSpline(x, y, lambda);
}

smoothingSpline::~smoothingSpline()
{

}

double smoothingSpline::evaluate(const double x)
{
  double dx = x - breaks(0);
  int i;
  for (i = 1; i < breaks.size(); i ++)
  {
    if (x - breaks(i) <= 0)
      break;
    dx = x - breaks(i);
  }      
  double val = coeffs(i-1,0)*dx*dx*dx + coeffs(i-1,1)*dx*dx + coeffs(i-1,2)*dx + coeffs(i-1,3);
  return val;
}

Eigen::VectorXd smoothingSpline::evaluate(const Eigen::VectorXd &x)
{
  Eigen::VectorXd y = x;
  for (int i = 0; i < x.size(); i++)
  {
    y(i) = evaluate(x(i));
  }
  return y;
}

void smoothingSpline::createSpline(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double lambda)
{
  Eigen::DenseIndex n = y.size() - 1;
  Eigen::VectorXd h = x.segment(1, n) - x.segment(0, n);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n-1, n-1);

  R(0, 0) = 2*(h(0) + h(1));
  for (int i = 1; i < n-2; i++)
  {
    R(i, i) = 2*(h(i) + h(i+1));
    R(i-1, i) = h(i);
    R(i, i-1) = h(i);
  }

  Eigen::VectorXd r = 3 / h.array();

  Eigen::MatrixXd Qt = Eigen::MatrixXd::Zero(n-1, n+1);
  for (int i = 0; i < n-1; i++)
  {
    Qt(i, i) = r(i);
    Qt(i, i+1) = -(r(i) + r(i+1));
    Qt(i, i+2) = r(i+1);
  }

  // weights are just the identity matrix
  Eigen::MatrixXd E = Eigen::MatrixXd::Identity(n+1, n+1);

  double mu = 2*(1-lambda)/(3*lambda);

  Eigen::MatrixXd A = mu * Qt * E * Qt.transpose() + R;
  Eigen::VectorXd B = Qt * y;

  Eigen::VectorXd b_1 = A.ldlt().solve(B); // A is guaranteed to be symmetric with 5 diagonal bands
  Eigen::VectorXd b = Eigen::VectorXd::Zero(n+1);
  b.segment(1,n-1) = b_1;

  Eigen::VectorXd d = y - mu * E * Qt.transpose() * b_1;
  Eigen::VectorXd a = (b.segment(1,n) - b.head(n)).array() / (3*h.array());
  // NOTE: There is a typo in the referenced paper for finding the c coefficient.
  //       The formula here is correct.
  Eigen::VectorXd c = (d.segment(1, n) - d.head(n)).array() / h.array() - 1.0 / 3.0 * (b.segment(1, n) + 2 * b.head(n)).array() * h.array();

  coeffs = Eigen::MatrixXd::Zero(n, 4);
  coeffs.col(0) = a;
  coeffs.col(1) = b.head(n);
  coeffs.col(2) = c;
  coeffs.col(3) = d.head(n);

  breaks = x;
}
