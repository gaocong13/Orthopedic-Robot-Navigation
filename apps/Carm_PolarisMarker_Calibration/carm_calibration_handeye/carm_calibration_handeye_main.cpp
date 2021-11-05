
// STD
#include <iostream>
#include <vector>

#include <fmt/format.h>

#include "xregProgOptUtils.h"
#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregLandmarkMapUtils.h"
#include "xregAnatCoordFrames.h"
#include "xregH5ProjDataIO.h"
#include "xregPnPUtils.h"
#include "xregHDF5.h"
#include "xregCIOSFusionDICOM.h"

#include "xregPAODrawBones.h"
#include "xregRigidUtils.h"

#include "bigssMath.h"

using namespace xreg;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;
constexpr bool kSAVE_REGI_DEBUG = true;

using size_type = std::size_t;

using Pt3         = Eigen::Matrix<CoordScalar,3,1>;
using Pt2         = Eigen::Matrix<CoordScalar,2,1>;

struct RegiErrorInfo
{
    CoordScalar rot_error_deg;
    CoordScalar trans_error;

    CoordScalar rot_x_error_deg;
    CoordScalar rot_y_error_deg;
    CoordScalar rot_z_error_deg;

    CoordScalar trans_x_error;
    CoordScalar trans_y_error;
    CoordScalar trans_z_error;
};

void ExportRegiErrorToDisk(FrameTransform regi_err_xform, const std::string file_name, const std::string exp_ID)
{
  RegiErrorInfo err;
  std::tie(err.rot_x_error_deg, err.rot_y_error_deg, err.rot_z_error_deg, err.trans_x_error, err.trans_y_error, err.trans_z_error) = RigidXformToEulerXYZAndTrans(regi_err_xform);
  err.rot_x_error_deg *= kRAD2DEG; err.rot_y_error_deg *= kRAD2DEG; err.rot_z_error_deg *= kRAD2DEG;

  std::ofstream regi_err;
  regi_err.open(file_name, std::ios::app);
  regi_err << "ExpID"  << ',' << exp_ID
           << ',' << err.trans_x_error       << ',' << err.trans_y_error       << ',' << err.trans_z_error
           << ',' << err.rot_x_error_deg     << ',' << err.rot_y_error_deg     << ',' << err.rot_z_error_deg    << '\n';
  regi_err.close();
}

FrameTransform ConvertSlicerToITK(std::vector<float> slicer_vec){
  FrameTransform RAS2LPS;
  RAS2LPS(0, 0) = -1; RAS2LPS(0, 1) = 0; RAS2LPS(0, 2) = 0; RAS2LPS(0, 3) = 0;
  RAS2LPS(1, 0) = 0;  RAS2LPS(1, 1) = -1;RAS2LPS(1, 2) = 0; RAS2LPS(1, 3) = 0;
  RAS2LPS(2, 0) = 0;  RAS2LPS(2, 1) = 0; RAS2LPS(2, 2) = 1; RAS2LPS(2, 3) = 0;
  RAS2LPS(3, 0) = 0;  RAS2LPS(3, 1) = 0; RAS2LPS(3, 2) = 0; RAS2LPS(3, 3) = 1;

  FrameTransform ITK_xform;
  for(size_type idx=0; idx<4; ++idx)
  {
    for(size_type idy=0; idy<3; ++idy)
    {
      ITK_xform(idy, idx) = slicer_vec[idx*3+idy];
    }
    ITK_xform(3, idx) = 0.0;
  }
  ITK_xform(3, 3) = 1.0;

  ITK_xform= RAS2LPS * ITK_xform * RAS2LPS;

  float tmp1, tmp2, tmp3;
  tmp1 = ITK_xform(0,0)*ITK_xform(0,3) + ITK_xform(0,1)*ITK_xform(1,3) + ITK_xform(0,2)*ITK_xform(2,3);
  tmp2 = ITK_xform(1,0)*ITK_xform(0,3) + ITK_xform(1,1)*ITK_xform(1,3) + ITK_xform(1,2)*ITK_xform(2,3);
  tmp3 = ITK_xform(2,0)*ITK_xform(0,3) + ITK_xform(2,1)*ITK_xform(1,3) + ITK_xform(2,2)*ITK_xform(2,3);

  ITK_xform(0,3) = -tmp1;
  ITK_xform(1,3) = -tmp2;
  ITK_xform(2,3) = -tmp3;

  return ITK_xform;
}

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Compute Hand-eye calibration of the C-arm customized fiducial to C-arm registration source frame.");
  po.set_arg_usage("<PnP xform folder> <calibration data folder> <result folder> <exp ID file>");
  po.set_min_num_pos_args(3);
  po.add("save-handeye", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "save-handeye", "Save Handeye Results")
  << false;
  po.add("compute-residual-error", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "compute-residual-error", "Compute residual error using the calibration result.")
  << false;

  po.add_backend_flags();

  try
  {
    po.parse(argc, argv);
  }
  catch (const ProgOpts::Exception& e)
  {
    std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
    po.print_usage(std::cerr);
    return kEXIT_VAL_BAD_USE;
  }

  if (po.help_set())
  {
    po.print_usage(std::cout);
    po.print_help(std::cout);
    return kEXIT_VAL_SUCCESS;
  }

  const bool verbose = po.get("verbose");
  std::ostream& vout = po.vout();

  const bool save_handeye = po.get("save-handeye");
  const bool compute_residual_error = po.get("compute-residual-error");

  const std::string root_pnp_folder           = po.pos_args()[0]; // Folder containing pnp xform
  const std::string root_cali_data_folder     = po.pos_args()[1]; // Folder containing calibration tracker xforms
  const std::string result_folder             = po.pos_args()[2]; // Folder to save results
  const std::string exp_list_path             = po.pos_args()[3]; // Source exp list file

  std::vector<std::string> exp_ID_list;
  int lineNumber = 0;
  /* Read exp ID list from file */
  {
    std::ifstream expIDFile(exp_list_path);
    // Make sure the file is open
    if(!expIDFile.is_open()) throw std::runtime_error("Could not open exp ID file");

    std::string line, csvItem;

    while(std::getline(expIDFile, line)){
      std::istringstream myline(line);
      while(getline(myline, csvItem)){
          exp_ID_list.push_back(csvItem);
      }
      lineNumber++;
    }
  }

  if(lineNumber!=exp_ID_list.size()) throw std::runtime_error("Exp ID list size mismatch!!!");

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  std::vector <vctFrm4x4> A_frames;    //< Transformation of A frames
  std::vector <vctFrm4x4> B_frames;    //< Transformation of B frames

  FrameTransformList pnp_xform_list;
  FrameTransformList RB4_wrt_CarmFid_xform_list;
  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string exp_ID = exp_ID_list[idx];

    const std::string pnp_xform_path = root_pnp_folder + "/pnp_xform" + exp_ID + ".h5";
    auto pnp_xform = ReadITKAffineTransformFromFile(pnp_xform_path);
    pnp_xform_list.push_back(pnp_xform);

    const std::string RB4_xform_path        = root_cali_data_folder + "/" + exp_ID + "/RB4.h5";
    H5::H5File h5_RB4(RB4_xform_path, H5F_ACC_RDWR);

    H5::Group RB4_transform_group           = h5_RB4.openGroup("TransformGroup");
    H5::Group RB4_group0                    = RB4_transform_group.openGroup("0");
    std::vector<float> RB4_slicer           = ReadVectorH5Float("TranformParameters", RB4_group0);

    FrameTransform RB4_xform = ConvertSlicerToITK(RB4_slicer);

    const std::string CarmFid_xform_path    = root_cali_data_folder + "/" + exp_ID + "/BayviewSiemensCArm.h5";
    H5::H5File h5_CarmFid(CarmFid_xform_path, H5F_ACC_RDWR);

    H5::Group CarmFid_transform_group       = h5_CarmFid.openGroup("TransformGroup");
    H5::Group CarmFid_group0                = CarmFid_transform_group.openGroup("0");
    std::vector<float> CarmFid_slicer       = ReadVectorH5Float("TranformParameters", CarmFid_group0);

    FrameTransform CarmFid_xform = ConvertSlicerToITK(CarmFid_slicer);

    FrameTransform RB4_wrt_CarmFid = RB4_xform.inverse() * CarmFid_xform;
    RB4_wrt_CarmFid_xform_list.push_back(RB4_wrt_CarmFid);

    vctFrm4x4 A_frame;
    vctFrm4x4 B_frame;

    for(size_type idx=0; idx<4; ++idx)
    {
      for(size_type idy=0; idy<4; ++idy)
      {
        A_frame[idx][idy] = RB4_wrt_CarmFid(idx, idy);
        B_frame[idx][idy] = pnp_xform(idx, idy);
      }
    }

    A_frames.push_back(A_frame);
    B_frames.push_back(B_frame);
  }

  if(A_frames.size() <= 5){
    std::cerr << "At least 5 frames are required for hand-eye calibration" << std::endl;
    return kEXIT_VAL_BAD_USE;
  }

  vctDoubleMat AX, BX, AY, BY;
  AX.SetSize(2*A_frames.size()*(A_frames.size()-1), 4);
  BX.SetSize(2*B_frames.size()*(B_frames.size()-1), 4);
  AY.SetSize(2*A_frames.size()*(A_frames.size()-1), 4);
  BY.SetSize(2*B_frames.size()*(B_frames.size()-1), 4);

  // precalculate inverse to solve for X
  unsigned int count = 0;
  for (unsigned int i=0; i<A_frames.size()-1; i++)
  {
    for (unsigned int j=i+1; j<A_frames.size(); j++)
    {
      AX.Ref(4, 4, 4*count, 0).Assign ( vctDoubleMat (A_frames[j].Inverse() * A_frames[i]));
      BX.Ref(4, 4, 4*count, 0).Assign ( vctDoubleMat (B_frames[j].Inverse() * B_frames[i]));
      AY.Ref(4, 4, 4*count, 0).Assign ( vctDoubleMat (A_frames[j] * A_frames[i].Inverse()));
      BY.Ref(4, 4, 4*count, 0).Assign ( vctDoubleMat (B_frames[j] * B_frames[i].Inverse()));
      ++count;
    }
  }


  vctFrm4x4 X, Y;
  BIGSS::ax_xb(AX, BX, X);
  BIGSS::ax_xb(AY, BY, Y);

  // Save hand-eye result to file
  FrameTransform handeye_X = FrameTransform::Identity();
  FrameTransform handeye_Y = FrameTransform::Identity();

  for(size_type idx=0; idx<4; ++idx)
  {
    for(size_type idy=0; idy<4; ++idy)
    {
      handeye_X(idx, idy) = X[idx][idy];
      handeye_Y(idx, idy) = Y[idx][idy];
    }
  }

  /*
  // For debugging purpose
  std::cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << std::endl;
  for (unsigned int i=0; i<A_frames.size()-1; i++)
  {
    for (unsigned int j=i+1; j<A_frames.size(); j++)
    {
      std::cout << "---------------------------------------" << std::endl;
      std::cout << "i: " << i << " j: " << j << std::endl;
      auto AX_xform = RB4_wrt_CarmFid_xform_list[j].inverse() * RB4_wrt_CarmFid_xform_list[i];
      auto XB_xform = pnp_xform_list[j].inverse() * pnp_xform_list[i];
      FrameTransform AX_X = AX_xform * handeye_X;
      FrameTransform X_XB = handeye_X * XB_xform;
      std::cout << "AX_X:" << std::endl << AX_X.matrix() << std::endl;
      std::cout << "X_XB:" << std::endl << X_XB.matrix() << std::endl;
    }
  }
  */

  if (save_handeye)
  {
    const std::string handeye_regi_Xfile = result_folder + "/handeye_X.h5";
    WriteITKAffineTransform(handeye_regi_Xfile, handeye_X);

    const std::string handeye_regi_Yfile = result_folder + "/handeye_Y.h5";
    WriteITKAffineTransform(handeye_regi_Yfile, handeye_Y);
  }

  Pt3 origin(0.0, 0.0, 0.0);

  if(compute_residual_error)
  {
    std::vector<float> residual_error_list;
    int count = 0;
    for(int idx = 0; idx < lineNumber - 1; ++idx)
    {
      for(int idy = idx+1; idy < lineNumber; ++idy)
      {
        FrameTransform rel_pnp_xform = pnp_xform_list[idx].inverse() * pnp_xform_list[idy];
        FrameTransform rel_fidcal_xform = handeye_X.inverse() * RB4_wrt_CarmFid_xform_list[idx].inverse() * RB4_wrt_CarmFid_xform_list[idy] * handeye_X;
        FrameTransform rel_residual_xform = rel_fidcal_xform.inverse() * rel_pnp_xform;
        ExportRegiErrorToDisk(rel_residual_xform, result_folder + "/carm_reposition_residual_error.txt", fmt::format("{:04d}", count));
        count++;

        Pt3 origin_Y_src = rel_pnp_xform * origin;
        Pt3 origin_X_src = rel_fidcal_xform * origin;
        Pt3 origin_diff = origin_Y_src - origin_X_src;
        residual_error_list.push_back(origin_diff.norm());
      }
    }

    float mean = std::accumulate(residual_error_list.begin(), residual_error_list.end(), 0.0) / residual_error_list.size();
    float sq_sum = std::inner_product(residual_error_list.begin(), residual_error_list.end(), residual_error_list.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / residual_error_list.size() - mean * mean);

    vout << "mean C-arm reposition residual error: " <<  mean << " +/- " << stdev << " mm" << std::endl;
  }

  if(compute_residual_error)
  {
    std::vector<float> residual_error_list;
    for(int idx = 0; idx < lineNumber; ++idx)
    {
      FrameTransform AX_xform = RB4_wrt_CarmFid_xform_list[idx] * handeye_X;
      FrameTransform YB_xform = handeye_Y * pnp_xform_list[idx];
      FrameTransform rel_residual_xform = AX_xform.inverse() * YB_xform;
      ExportRegiErrorToDisk(rel_residual_xform, result_folder + "/AXYB_residual_error.txt", fmt::format("{:02d}", idx));

      Pt3 origin_Y_src = YB_xform * origin;
      Pt3 origin_X_src = AX_xform * origin;
      Pt3 origin_diff = origin_Y_src - origin_X_src;
      residual_error_list.push_back(origin_diff.norm());
    }

    float mean = std::accumulate(residual_error_list.begin(), residual_error_list.end(), 0.0) / residual_error_list.size();
    float sq_sum = std::inner_product(residual_error_list.begin(), residual_error_list.end(), residual_error_list.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / residual_error_list.size() - mean * mean);

    vout << "mean AX YB residual error: " << mean << " +/- " << stdev << " mm" << std::endl;
  }

  // print out results
  std::cout << "handeye X = " << std::endl << X << std::endl;
  std::cout << "handeye Y = " << std::endl << Y << std::endl;

  return kEXIT_VAL_SUCCESS;
}
