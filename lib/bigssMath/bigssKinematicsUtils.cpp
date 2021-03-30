
#include "bigssKinematicsUtils.h"

vctMatrix4x4 FrameInverse(const vctMatrix4x4& f)
{
  vctMatrix4x4 f_inv = vctMatrix4x4::Eye();
  
  vctDynamicConstMatrixRef<double> R(f, 0, 0, 3, 3);
  vctDynamicMatrixRef<double> R_T(f_inv, 0, 0, 3, 3);
  
  R_T = R.Transpose();
  
  vctDynamicConstMatrixRef<double> trans(f, 0, 3, 3, 1);
  
  // inverse component of translation
  vctDynamicMatrixRef<double>(f_inv, 0, 3, 3, 1) = -1.0 * R_T * trans;
  
  return f_inv;
}

vctFrm4x4 ConvertFromURAxisAnglePose(const double ur_pose[6], const bool convert_from_m)
{
  vctFrm4x4 pose;
  
  pose.Translation().Assign(ur_pose[0],
                            ur_pose[1],
                            ur_pose[2]);
  if (convert_from_m)
  {
    pose.Translation() *= 1000.0;
  }
  
  pose.Rotation().FromNormalized(
      vctRodriguezRotation3<double>(ur_pose[3], ur_pose[4], ur_pose[5]));

  return pose;
}

void ConvertToURAxisAnglePose(const vctFrm4x4& src, double ur_pose[6], const bool convert_to_m)
{
  ur_pose[0] = src.Translation().X();
  ur_pose[1] = src.Translation().Y();;
  ur_pose[2] = src.Translation().Z();
  
  if (convert_to_m)
  {
    ur_pose[0] /= 1000.0;
    ur_pose[1] /= 1000.0;
    ur_pose[2] /= 1000.0;
  }
  
  vctAxAnRot3 axis_angle(src.Rotation());
  
  vctFixedSizeVectorRef<double,3,1> ur_ax_ang(ur_pose + 3);
  ur_ax_ang  = axis_angle.Axis();
  ur_ax_ang *= axis_angle.Angle();
}


