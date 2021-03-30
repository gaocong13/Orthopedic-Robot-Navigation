#ifndef _interpolatingSpline_h
#define _interpolatingSpline_h

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "spline.h"

namespace {
  namespace tk {
    class spline;
  }
}


/*!
 * This defines a set of routines to construct and act on interpolating
 * splines. The splines are modeled as cubic, and the knot points are maintained.
 * Note that this is different from the smoothing spline!
 *
 * In particular, use methods from this class to define a spline and perform
 * actions on it, such as identifying the intersection of a spline with a line.
 *
 * The spline.h file forms the basis for this.
 */
namespace BIGSS {
  /// Structure for a 3d spline representation
  struct spline3d {
    ::tk::spline *x;  //< Spline representation of the X lunate trace points
    ::tk::spline *y;  //< Spline representation of the Y lunate trace points
    ::tk::spline *z;  //< Spline representation of the Z lunate trace points

    unsigned int pieces;       //< The number of pieces in the spline (nPts - 1)
  };

  /// Construct the splines from the specific points
  ///   A spline is constructed for x, y, z.
  /// \param pts The points to create a spline for.
  /// \param s The resulting spline
  void constructSpline (const Eigen::Matrix3Xd &pts, struct spline3d &s);

  /// Find the intersection of a spline curve with a plane
  /// \param s The 3d spline curve
  /// \param N The plane definition
  /// \param Rn The permissible ray direction of the intersection
  /// \param cen The center point on the plane
  void findSplineIntersection (const struct spline3d s, const Eigen::Vector4d &N, const Eigen::Vector3d &Rn, const Eigen::Vector3d &cen, Eigen::Matrix3Xd &pts);
}
#endif // _interpolatingSpline_h
