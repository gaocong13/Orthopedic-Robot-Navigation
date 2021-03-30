template<typename _Matrix_Type_>
static void pinv(const _Matrix_Type_ &a, _Matrix_Type_ &result, double epsilon = std::numeric_limits<typename _Matrix_Type_::Scalar>::epsilon())
{
  _Matrix_Type_ aCopy = a;
  if(a.rows()<a.cols())
    aCopy = a.transpose();

  Eigen::JacobiSVD< _Matrix_Type_ > svd = aCopy.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

  typename _Matrix_Type_::Scalar tolerance = epsilon * std::max(aCopy.cols(), aCopy.rows()) * svd.singularValues()[0];
  
  result = svd.matrixV() * _Matrix_Type_( (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().
      array().inverse(), 0) ).asDiagonal() * svd.matrixU().adjoint();

  if(a.rows()<a.cols())
    result = result.transpose();
}
