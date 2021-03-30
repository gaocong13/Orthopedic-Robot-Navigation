
#ifndef _BIGSSKINEMATICSUTILS_H_
#define _BIGSSKINEMATICSUTILS_H_

#include <vector>
#include <cmath>

#include <cisstCommon.h>
#include <cisstVector.h>

typedef vctFixedSizeMatrix<double,4,4> vctMatrix4x4;
typedef vctFixedSizeMatrix<double,6,6> vctMatrix6x6;
typedef vctFixedSizeVector<double,6> vctVector6;

//inline vctFrm4x4 RotX(const double theta)
//{
//  vctFrm4x4 r;
//  r.Rotation().From( vctAxAnRot3(vct3(1.0, 0.0, 0.0), theta) );
//
//  return r;
//}
//
//inline vctFrm4x4 RotY(const double theta)
//{
//  vctFrm4x4 r;
//  r.Rotation().From( vctAxAnRot3(vct3(0.0, 1.0, 0.0), theta) );
//
//  return r;
//}
//
//inline vctFrm4x4 RotZ(const double theta)
//{
//  vctFrm4x4 r;
//  r.Rotation().From( vctAxAnRot3(vct3(0.0, 0.0, 1.0), theta) );
//
//  return r;
//}
//
//inline vctFrm4x4 TransXYZ(const double x, const double y, const double z)
//{
//  vctFrm4x4 r;
//  r.Translation().Assign(x, y, z);
//
//  return r;
//}

vctMatrix4x4 FrameInverse(const vctMatrix4x4& f);

/// \brief Converts a UR pose double array into a CISST 4x4 homogeneous matrix
/// \param ur_pose The source UR pose double array
/// \param convert_from_m (optional) Flag indicating the the UR translation component
///                       should be converted from meters into mm (defaults to false)
/// \return A 4x4 CISST frame
vctFrm4x4 ConvertFromURAxisAnglePose(const double ur_pose[6], const bool convert_from_m = false);

/// \brief Converts a CISST frame into a UR pose double array
/// \param src The source CISST frame
/// \param ur_pose The destination UR pose array
/// \param convert_to_m (optional) Flag indicating that the UR translation component
///                     should be converted into meters into mm (defaults to false)
void ConvertToURAxisAnglePose(const vctFrm4x4& src, double ur_pose[6], const bool convert_to_m = false);

#endif
