#ifndef _smoothingSpline_h
#define _smoothingSpline_h

#include <Eigen/Core>

/*!
 * This class defines a smoothing spline. This works similar to 
 * Matlab's smoothingspline fit. 
 *
 * This constructs a cubic smoothing spline of the form
 *  S_i(x) = a_i * (x-x_i)^3 + b_i * (x-x_i)^2 + c_i * (x-x_i) + d_i
 *
 * The implementation is based off of
 * http://www.physics.muni.cz/~jancely/NM/Texty/Numerika/CubicSmoothingSpline.pdf
 */
namespace BIGSS {
  class smoothingSpline
  {
    // attributes
    Eigen::MatrixXd coeffs; //!< The coefficients, [a b c d]
    Eigen::VectorXd breaks; //!< The break points

    // methods
  public:

    /// Construct a smoothing spline
    /// \param y The y-values of the (equally-spaced) curve    
    /// \param lambda The smoothing parameter for the spline
    smoothingSpline(const Eigen::VectorXd &y, const double lambda = 0.999999523162842);
    
    /// Construct a smoothing spline
    /// \param x The x-values parameterizing the curve (typically, this is for 1D data, but x defines the spacing)
    /// \param y The y-values of the curve
    /// \param lambda The smoothing parameter for the spline
    smoothingSpline(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double lambda = 0.999999523162842);
    ~smoothingSpline();

    /// Evaluate the spline at a point x
    double evaluate(const double x);

    /// Evaluate the spline at a set of points x
    Eigen::VectorXd evaluate(const Eigen::VectorXd &x);


  protected:
    /// Create the actual spline
    void createSpline(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const double lambda);
  };

}

    

#endif // _smoothingSpline_h
