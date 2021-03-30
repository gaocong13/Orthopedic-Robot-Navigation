#include "interpolatingSpline.h"

#include <unsupported/Eigen/Polynomials>

using namespace BIGSS;

void
BIGSS::constructSpline (const Eigen::Matrix3Xd &pts, struct spline3d &s)
{
  size_t n = pts.cols();

  std::vector <double> id(n);
  std::vector <double> x(n);
  std::vector <double> y(n);
  std::vector <double> z(n);

  for (size_t i=0; i<n; i++)
  {
    id[i] = static_cast<double>(i);
    x[i] = pts.col(i).x();
    y[i] = pts.col(i).y();
    z[i] = pts.col(i).z();
  }

  s.x->set_points (id, x);
  s.y->set_points (id, y);
  s.z->set_points (id, z);

  s.pieces = static_cast<unsigned int>(n) - 1;
}

void 
BIGSS::findSplineIntersection (const struct spline3d s, const Eigen::Vector4d &N, const Eigen::Vector3d &Rn, const Eigen::Vector3d &cen, Eigen::Matrix3Xd &pts)
{
  //bool checkDirection = Rn.AlmostEqual();
  bool checkDirection = !N.segment<3>(0).isApprox(Rn);


  Eigen::PolynomialSolver<double, 3> solver;
  Eigen::Vector4d eqn;
  Eigen::PolynomialSolver<double, 3>::RootsType r;

  double a, b, c;
  double x, y, z;
  for (unsigned int i=0; i<s.pieces; i++)
  {
    eqn.setZero ();
    x = (*s.x)(i);
    y = (*s.y)(i);
    z = (*s.z)(i);

    s.x->get_coeffs(i, a, b, c);
    eqn[0] += N.x() * a;
    eqn[1] += N.x() * b;
    eqn[2] += N.x() * c;
    eqn[3] += N.x() * x;

    s.y->get_coeffs(i, a, b, c);
    eqn[0] += N.y() * a;
    eqn[1] += N.y() * b;
    eqn[2] += N.y() * c;
    eqn[3] += N.y() * y;
    
    s.z->get_coeffs(i, a, b, c);
    eqn[0] += N.z() * a;
    eqn[1] += N.z() * b;
    eqn[2] += N.z() * c;
    eqn[3] += N.z() * z;

    eqn[3] += N.w();

    // switch ordering to work with Eigen
    solver.compute (eqn.reverse());
    //solver.compute(eqn);
    r = solver.roots();

    // find the matching root(s), if any
    for (int j=0; j<3; j++)
    {
      // check if real root
      if (std::abs(r[j].imag()) < 0.000001)
      {
        // check if valid segment length, if so, add to list of intersection points
        // [in this implementation, we know that segment length == 1 by construction]
        double root = r[j].real();
        if ((root >= 0) && (root <= 1))
        {
          int currCol = pts.cols();
          x = (*s.x)(i + root);
          y = (*s.y)(i + root);
          z = (*s.z)(i + root);
          if (checkDirection)
          {
            Eigen::Vector3d pt(x,y,z);
            if (Rn.dot(pt - cen) > 0) {
              pts.conservativeResize(Eigen::NoChange, pts.cols() + 1);
              pts.col(currCol) << x, y, z;
            }
          } else {
            pts.conservativeResize(Eigen::NoChange, pts.cols() + 1);
            pts.col(currCol) << x, y, z;
          }
        }
      }
    }
  }
}
