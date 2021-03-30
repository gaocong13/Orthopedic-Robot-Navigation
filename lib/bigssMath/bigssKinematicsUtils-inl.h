
template <unsigned _NumJoints>
vctFrm4x4 DHKinematics<_NumJoints>::dhf(const size_type i, const double theta) const
{
  CMN_ASSERT(i < kNUM_JOINTS);
  return RotZ(theta) * TransXYZ(0, 0, d_[i]) * TransXYZ(a_[i], 0, 0) * RotX(alpha_[i]);
}

template <unsigned _NumJoints>
vctMatrix4x4 DHKinematics<_NumJoints>::pose(const JointVector& q, const size_type i) const
{
  vctMatrix4x4 T_i_to_0 = vctMatrix4x4::Eye();

  for (size_type j = 0; j < i; ++j)
  {
    T_i_to_0 = T_i_to_0 * dhf(j, q[j]);
  }

  return T_i_to_0;
}

template <unsigned _NumJoints>
typename DHKinematics<_NumJoints>::JacobianMatrix
DHKinematics<_NumJoints>::geo_jac(const JointVector& q, const vct3 offset) const
{

  // This implements the geometric Jacobian as described by Spong
  // for revolute-only joints with a DH parameterization

  typedef vctDynamicMatrixRef<double> vctJacMatrixRef;
  JacobianMatrix jac;

  std::vector<vct3> origins_in_0(kNUM_JOINTS + 1, vct3(0.0));

  vctMatrix4x4 T_i_to_0 = vctMatrix4x4::Eye();

  for (size_type i = 1; i <= kNUM_JOINTS; ++i)
  {
    // eq 4.59, 4.47 in Spong (pages 133, 131)
    vctJacMatrixRef(jac, 3, i-1, 3, 1) = vctJacMatrixRef(T_i_to_0, 0, 2, 3, 1);  // pulls out the z-axis

    T_i_to_0 = T_i_to_0 * dhf(i-1, q[i-1]);

    vct3& cur_origin = origins_in_0[i];    
    cur_origin(0) = T_i_to_0(0,3);
    cur_origin(1) = T_i_to_0(1,3);
    cur_origin(2) = T_i_to_0(2,3);
  }

  // Modification due to offset parameter
  vctFrm3 T_i_to_0_Frm3;
  T_i_to_0_Frm3.FromNormalized((vctFrm4x4)(T_i_to_0));
  origins_in_0.at(kNUM_JOINTS) = T_i_to_0_Frm3 * offset;

  vct3 tmp_vec;
  for (unsigned long i = 1; i <= kNUM_JOINTS; ++i)
  {
    // eq 4.57 in Spong (page 133)

    // there is no convenient shortcut for extracting a vector from a
    // CISST matrix AND have it inherit vector operations, such as cross product
    tmp_vec(0) = jac(3, i-1);
    tmp_vec(1) = jac(4, i-1);
    tmp_vec(2) = jac(5, i-1);

    // % is shorthand for cross product
    tmp_vec = tmp_vec % (origins_in_0[kNUM_JOINTS] - origins_in_0[i-1]);

    jac(0, i-1) = tmp_vec(0);
    jac(1, i-1) = tmp_vec(1);
    jac(2, i-1) = tmp_vec(2);
  }

  return jac;

}
