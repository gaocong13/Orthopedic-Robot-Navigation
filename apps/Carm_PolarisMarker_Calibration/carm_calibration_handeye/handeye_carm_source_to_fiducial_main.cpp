// STD
#include <iostream>
#include <vector>

#include <fmt/format.h>

// jhmr
#include "jhmrMultiObjMultiLevel2D3DRegi.h"
#include "jhmrCameraRayCastingGPU.h"
#include "jhmrCIOSFusionDICOM.h"
#include "jhmrLandmark2D3DRegi.h"

#include "jhmrProgOptUtils.h"
#include "jhmrHDF5.h"
#include "jhmrPAOUtils.h"
#include "jhmrProjDataH5.h"
#include "jhmrFCSVUtils.h"
#include "jhmrAnatCoordFrames.h"
#include "jhmrPAOCutsXML.h"
#include "jhmrSpline.h"
#include "jhmrRegi2D3DDebugH5.h"
#include "jhmrRegi2D3DDebugIO.h"

#include "bigssMath.h"

using namespace jhmr;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;

using size_type = std::size_t;

using Pt3         = Eigen::Matrix<CoordScalar,3,1>;
using Pt2         = Eigen::Matrix<CoordScalar,2,1>;

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
  typedef Landmark2D3DRegi::CamModel CamModel;
  ProgOpts po;

  jhmrPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Compute Hand-eye calibration of the C-arm customized fiducial to C-arm registration source frame.");
  po.set_arg_usage("<PnP xform folder> <calibration data folder> <result folder> <exp ID file>");
  po.set_min_num_pos_args(3);
  po.add("save-handeye", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "save-handeye", "Save Handeye Results")
  << false;
  po.add("compute-residual-error", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "compute-residual-error", "Compute residual error using the calibration result.")
  << false;
  
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
 
  const bool save_handeye = po.get("save-handeye");
  const bool compute_residual_error = po.get("compute-residual-error");
  
  const std::string root_pnp_folder           = po.pos_args()[0]; // Folder containing pnp xform
  const std::string root_cali_data_folder     = po.pos_args()[1]; // Folder containing calibration tracker xforms
  const std::string result_folder             = po.pos_args()[2]; // Folder to save results
  const std::string exp_list_path             = po.pos_args()[3]; // Source exp list file
  
  const CamModel default_cam = NaiveCamModelFromCIOSFusion(
  MakeNaiveCIOSFusionMetaDR(), true).cast<CoordScalar>();
  
  std::vector <vctFrm4x4> A_frames;    //< Transformation of A frames
  std::vector <vctFrm4x4> B_frames;    //< Transformation of B frames
  
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
  
  FrameTransformList pnp_xform_list;
  FrameTransformList RB4_wrt_CarmFid_xform_list;
  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string exp_ID = exp_ID_list[idx];
    
    const std::string pnp_xform_path = root_pnp_folder + "/pnp_xform" + exp_ID + ".h5";
    FrameTransform pnp_xform;
    ReadITKAffineTransformFromFile(pnp_xform_path, &pnp_xform);
    pnp_xform = pnp_xform.inverse();
    pnp_xform_list.push_back(pnp_xform);
    
    const std::string RB4_xform_path        = root_cali_data_folder + "/" + exp_ID + "/RB4.h5";
    H5::H5File h5_RB4(RB4_xform_path, H5F_ACC_RDWR);
    
    H5::Group RB4_transform_group           = h5_RB4.openGroup("TransformGroup");
    H5::Group RB4_group0                    = RB4_transform_group.openGroup("0");
    std::vector<float> RB4_slicer           = ReadVectorH5<float>("TranformParameters", RB4_group0);
    
    FrameTransform RB4_xform = ConvertSlicerToITK(RB4_slicer);
    
    const std::string CarmFid_xform_path    = root_cali_data_folder + "/" + exp_ID + "/BayviewSiemensCArm.h5";
    H5::H5File h5_CarmFid(CarmFid_xform_path, H5F_ACC_RDWR);
    
    H5::Group CarmFid_transform_group       = h5_CarmFid.openGroup("TransformGroup");
    H5::Group CarmFid_group0                = CarmFid_transform_group.openGroup("0");
    std::vector<float> CarmFid_slicer       = ReadVectorH5<float>("TranformParameters", CarmFid_group0);
    
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
    
    std::cout << exp_ID << std::endl;
    std::cout << "A frame:\n" << A_frame << std::endl;
    std::cout << "B frame:\n" << B_frame << std::endl;
    std::cout << " ***************************************** " << std::endl;
    
    A_frames.push_back(A_frame);
    B_frames.push_back(B_frame);
  }

  if(A_frames.size() <= 5){
    std::cerr << "At least 5 frames are required for hand-eye calibration" << std::endl;
    return kEXIT_VAL_BAD_USE;
  }
  
  vctDoubleMat AX, BX, AY, BY;
  AX.SetSize(2*A_frames.size()*(A_frames.size()+1), 4);
  BX.SetSize(2*B_frames.size()*(B_frames.size()+1), 4);
  AY.SetSize(2*A_frames.size()*(A_frames.size()+1), 4);
  BY.SetSize(2*B_frames.size()*(B_frames.size()+1), 4);
  
  // precalculate inverse to solve for X
  unsigned int count = 0;
  for (unsigned int i=0; i<A_frames.size(); i++)
  {
    for (unsigned int j=i; j<A_frames.size(); j++)
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

  // print out results
  std::cout << "X = " << std::endl << X << std::endl;
  std::cout << "Y = " << std::endl << Y << std::endl;

  // Save hand-eye result to file
  FrameTransform handeye_X = FrameTransform::Identity();
  FrameTransform handeye_Y = FrameTransform::Identity();
  Eigen::Matrix3d handeye_X_rot;
  for(size_type idx=0; idx<4; ++idx)
  {
    for(size_type idy=0; idy<4; ++idy)
    {
      handeye_X(idx, idy) = X[idx][idy];
      handeye_Y(idx, idy) = Y[idx][idy];
    }
  }
  
  for(size_type idx=0; idx<3; ++idx)
  {
    for(size_type idy=0; idy<3; ++idy)
    {
      handeye_X_rot(idx, idy) = X[idx][idy];
    }
  }
  
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
    for(int idx = 0; idx < lineNumber; ++idx)
    {
      Pt3 curcarm_source_wrt_refcarm_source = handeye_Y * pnp_xform_list[idx] * origin;
      Pt3 curcarm_fid_wrt_refcarm_fid = RB4_wrt_CarmFid_xform_list[idx] * handeye_X * origin;
      Pt3 carm_motion = curcarm_source_wrt_refcarm_source - curcarm_fid_wrt_refcarm_fid;
      residual_error_list.push_back(carm_motion.norm());
    }
    
    std::cout << "mean residual error:" << std::accumulate(residual_error_list.begin(), residual_error_list.end(), 0.0) / residual_error_list.size() << " mm" << std::endl;
  }
  
  return kEXIT_VAL_SUCCESS;
}


