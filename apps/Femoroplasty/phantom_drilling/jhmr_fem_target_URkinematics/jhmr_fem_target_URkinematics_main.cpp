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
#include "CommonTBME.h"
#include "jhmrRegi2D3DDebugH5.h"
#include "jhmrRegi2D3DDebugIO.h"

#include "bigssMath.h"


using namespace jhmr;
using namespace jhmr::tbme;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;

using size_type = std::size_t;

using Pt3         = Eigen::Matrix<CoordScalar,3,1>;
using Pt2         = Eigen::Matrix<CoordScalar,2,1>;

FrameTransform ConvertSlicerToITK(std::vector<float> slicer_vec)
{
  FrameTransform RAS2LPS;
  RAS2LPS(0, 0) = -1; RAS2LPS(0, 1) = 0; RAS2LPS(0, 2) = 0; RAS2LPS(0, 3) = 0;
  RAS2LPS(1, 0) = 0;  RAS2LPS(1, 1) = -1;RAS2LPS(1, 2) = 0; RAS2LPS(1, 3) = 0;
  RAS2LPS(2, 0) = 0;  RAS2LPS(2, 1) = 0; RAS2LPS(2, 2) = 1; RAS2LPS(2, 3) = 0;
  RAS2LPS(3, 0) = 0;  RAS2LPS(3, 1) = 0; RAS2LPS(3, 2) = 0; RAS2LPS(3, 3) = 1;
  
  FrameTransform Slicer_xform;
  for(size_type idx=0; idx<4; ++idx)
  {
    for(size_type idy=0; idy<3; ++idy)
    {
      Slicer_xform(idy, idx) = slicer_vec[idx*3+idy];
    }
    Slicer_xform(3, idx) = 0.0;
  }
  Slicer_xform(3, 3) = 1.0;

  FrameTransform ITK_xform = RAS2LPS * Slicer_xform * RAS2LPS;
  
  float tmp1, tmp2, tmp3;
  tmp1 = ITK_xform(0,0)*ITK_xform(0,3) + ITK_xform(0,1)*ITK_xform(1,3) + ITK_xform(0,2)*ITK_xform(2,3);
  tmp2 = ITK_xform(1,0)*ITK_xform(0,3) + ITK_xform(1,1)*ITK_xform(1,3) + ITK_xform(1,2)*ITK_xform(2,3);
  tmp3 = ITK_xform(2,0)*ITK_xform(0,3) + ITK_xform(2,1)*ITK_xform(1,3) + ITK_xform(2,2)*ITK_xform(2,3);
  
  ITK_xform(0,3) = -tmp1;
  ITK_xform(1,3) = -tmp2;
  ITK_xform(2,3) = -tmp3;
  
  return ITK_xform;
}

FrameTransform ConvertITKToSlicer(FrameTransform ITK_xform)
{
  FrameTransform Slicer_xform = FrameTransform::Identity();
  
  FrameTransform RAS2LPS;
  RAS2LPS(0, 0) = -1; RAS2LPS(0, 1) = 0; RAS2LPS(0, 2) = 0; RAS2LPS(0, 3) = 0;
  RAS2LPS(1, 0) = 0;  RAS2LPS(1, 1) = -1;RAS2LPS(1, 2) = 0; RAS2LPS(1, 3) = 0;
  RAS2LPS(2, 0) = 0;  RAS2LPS(2, 1) = 0; RAS2LPS(2, 2) = 1; RAS2LPS(2, 3) = 0;
  RAS2LPS(3, 0) = 0;  RAS2LPS(3, 1) = 0; RAS2LPS(3, 2) = 0; RAS2LPS(3, 3) = 1;
  
  ITK_xform(0,3) = -ITK_xform(0,3);
  ITK_xform(1,3) = -ITK_xform(1,3);
  ITK_xform(2,3) = -ITK_xform(2,3);
  
  FrameTransform ITK_R = FrameTransform::Identity();
  
  for(size_type idx=0; idx<3; ++idx)
  {
    for(size_type idy=0; idy<3; ++idy)
    {
      ITK_R(idx, idy) = ITK_xform(idx, idy);
    }
  }
  
  FrameTransform ITK_R_inv = ITK_R.inverse();
  
  float tmp1, tmp2, tmp3;
  tmp1 = ITK_R_inv(0,0)*ITK_xform(0,3) + ITK_R_inv(0,1)*ITK_xform(1,3) + ITK_R_inv(0,2)*ITK_xform(2,3);
  tmp2 = ITK_R_inv(1,0)*ITK_xform(0,3) + ITK_R_inv(1,1)*ITK_xform(1,3) + ITK_R_inv(1,2)*ITK_xform(2,3);
  tmp3 = ITK_R_inv(2,0)*ITK_xform(0,3) + ITK_R_inv(2,1)*ITK_xform(1,3) + ITK_R_inv(2,2)*ITK_xform(2,3);
  
  ITK_xform(0,3) = tmp1;
  ITK_xform(1,3) = tmp2;
  ITK_xform(2,3) = tmp3;
  
  FrameTransform Slicer_xform_before_R_trans = RAS2LPS.inverse() * ITK_xform * RAS2LPS.inverse();
  for(size_type idx=0; idx<3; ++idx)
  {
    for(size_type idy=0; idy<3; ++idy)
    {
      Slicer_xform(idy, idx) = Slicer_xform_before_R_trans(idx, idy);
    }
    Slicer_xform(idx, 3) = Slicer_xform_before_R_trans(idx, 3);
  }
  
  return Slicer_xform;
}

FrameTransform Get_needletip_wrt_target_xfrom(vct3 vct3_base_to_tip, vct3 vct3_entry_to_target, Pt3 NT_target_pt, Pt3 NT_needletip_pt)
{
  vct3 axis_vec =  vctCrossProduct(vct3_base_to_tip, vct3_entry_to_target);
  axis_vec = axis_vec / axis_vec.Norm();
  float angle = acos( vctDotProduct(vct3_base_to_tip, vct3_entry_to_target)/(vct3_base_to_tip.Norm() * vct3_entry_to_target.Norm()));
  
  vctAxAnRot3 axRot(axis_vec, angle);
  vctMatRot3 rotMat(axRot);
  
  // Initialize xform needletip_wrt_target
  FrameTransform needletip_wrt_target_xform = FrameTransform::Identity();
  for (size_type idx = 0; idx < 3; ++idx)
  {
    for (size_type idy = 0; idy < 3; ++idy)
    {
      needletip_wrt_target_xform(idx, idy) = rotMat[idx][idy];
    }
  }
  // translation part from needletip frame origin to target frame origin
  Pt3 NT_needletip_wrt_target = NT_target_pt - NT_needletip_pt;
  needletip_wrt_target_xform(0, 3) = NT_needletip_wrt_target[0];
  needletip_wrt_target_xform(1, 3) = NT_needletip_wrt_target[1];
  needletip_wrt_target_xform(2, 3) = NT_needletip_wrt_target[2];
  
  return needletip_wrt_target_xform;
}

FrameTransformList Target_Drillpose_from_Planning(const FrameTransform spine_xform, const FrameTransform drill_xform,
                                             const std::string planning_landmark_fcsv_path, const std::string needle_landmark_fcsv_path,
                                             const std::string ID_prefix)
{
  // Calculate xform drill w.r.t. spine
  FrameTransform drill_wrt_spine_xform;
  drill_wrt_spine_xform = drill_xform * spine_xform.inverse();
  
  std::cout << "reading entry & target landmarks from FCSV file..." << std::endl;
  auto planning_landmark_3dfcsv = ReadFCSVFileNamePtMap<Pt3>(planning_landmark_fcsv_path);
  FromMapConvertRASToLPS(planning_landmark_3dfcsv.begin(), planning_landmark_3dfcsv.end());
  
  std::cout << "reading drill needle landmarks from FCSV file..." << std::endl;
  auto needle_landmark_3dfcsv = ReadFCSVFileNamePtMap<Pt3>(needle_landmark_fcsv_path);
  FromMapConvertRASToLPS(needle_landmark_3dfcsv.begin(), needle_landmark_3dfcsv.end());
  
  FrameTransform needletip_wrt_drill = FrameTransform::Identity();
  Pt3 drill_needletip_pt;
  Pt3 drill_needlebase_pt;
  {
    auto drill_needletip_fcsv = needle_landmark_3dfcsv.find("NeedleTip");
    
    if (drill_needletip_fcsv != needle_landmark_3dfcsv.end()){
      drill_needletip_pt = drill_needletip_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND DRILL NEEDLE TIP PT" << std::endl;
    }
    
    auto drill_needlebase_fcsv = needle_landmark_3dfcsv.find("NeedleBase");
    
    if (drill_needlebase_fcsv != needle_landmark_3dfcsv.end()){
      drill_needlebase_pt = drill_needlebase_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND DRILL NEEDLE BASE PT" << std::endl;
    }
    
    needletip_wrt_drill.matrix()(0,3) = -drill_needletip_pt[0];
    needletip_wrt_drill.matrix()(1,3) = -drill_needletip_pt[1];
    needletip_wrt_drill.matrix()(2,3) = -drill_needletip_pt[2];
  }
  
  Pt3 base_pt;
  Pt3 safe_target_pt;
  Pt3 real_target_pt;
  {
    const std::string base_fd = "base" + ID_prefix;
    auto base_fcsv = planning_landmark_3dfcsv.find(base_fd);
    
    if (base_fcsv != planning_landmark_3dfcsv.end()){
      base_pt = base_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND BASE PT" << std::endl;
    }
    
    const std::string safe_target_fd = "safe-target" + ID_prefix;
    auto safe_target_fcsv = planning_landmark_3dfcsv.find(safe_target_fd);
    
    if (safe_target_fcsv != planning_landmark_3dfcsv.end()){
      safe_target_pt = safe_target_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND SAFE TARGET PT" << std::endl;
    }
    
    const std::string real_target_fd = "real-target" + ID_prefix;
    auto real_target_fcsv = planning_landmark_3dfcsv.find(real_target_fd);
    
    if (real_target_fcsv != planning_landmark_3dfcsv.end()){
      real_target_pt = real_target_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND REAL TARGET PT" << std::endl;
    }
  }
  
  const CamModel default_cam = NaiveCamModelFromCIOSFusion(
  MakeNaiveCIOSFusionMetaDR(), true).cast<CoordScalar>();
  
  FrameTransform needletip_ref_wrt_spine = needletip_wrt_drill * drill_wrt_spine_xform;
  
  // Transform needle tip from drill frame to needletip (NT) frame
  Pt3 NT_needletip_pt = needletip_wrt_drill * drill_needletip_pt;
  
  // Transform needle base from drill frame to needletip (NT) frame
  Pt3 NT_needlebase_pt = needletip_wrt_drill * drill_needlebase_pt;
  
  // Transform entry point from spine frame to needletip (NT) frame
  Pt3 NT_entry_pt = needletip_ref_wrt_spine * base_pt;
  
  // Transform safe target point from spine frame to needletip (NT) frame
  Pt3 NT_safe_target_pt = needletip_ref_wrt_spine * safe_target_pt;
  
  // Transform real target point from spine frame to needletip (NT) frame
  Pt3 NT_real_target_pt = needletip_ref_wrt_spine * real_target_pt;
  
  // Calculate vector from tip to base in needletip frame
  Pt3 NT_vec_base_to_tip = NT_needlebase_pt - NT_needletip_pt;
  
  // Calculate vector from entry to safe target point in needletip frame
  Pt3 NT_vec_entry_to_safe_target = NT_entry_pt - NT_safe_target_pt;
  
  // Calculate vector from entry to real target point in needletip frame
  Pt3 NT_vec_entry_to_real_target = NT_entry_pt - NT_real_target_pt;
  
  // Construct cisst vectors
  vct3 vct3_base_to_tip(NT_vec_base_to_tip[0], NT_vec_base_to_tip[1], NT_vec_base_to_tip[2]);
  vct3 vct3_entry_to_safe_target(NT_vec_entry_to_safe_target[0], NT_vec_entry_to_safe_target[1], NT_vec_entry_to_safe_target[2]);
  vct3 vct3_entry_to_real_target(NT_vec_entry_to_real_target[0], NT_vec_entry_to_real_target[1], NT_vec_entry_to_real_target[2]);
  
  // Get transformation from vector
  FrameTransform needletip_wrt_safe_target_xform = Get_needletip_wrt_target_xfrom(vct3_base_to_tip, vct3_entry_to_safe_target, NT_safe_target_pt, NT_needletip_pt);
  
  FrameTransform needletip_wrt_real_target_xform = Get_needletip_wrt_target_xfrom(vct3_base_to_tip, vct3_entry_to_real_target, NT_real_target_pt, NT_needletip_pt);
  
  // Compute final transformation
  FrameTransform drill_safe_target_wrt_cam = needletip_wrt_drill.inverse() * needletip_wrt_safe_target_xform.inverse() * needletip_wrt_drill * drill_xform;
  
  FrameTransform drill_real_target_wrt_cam = needletip_wrt_drill.inverse() * needletip_wrt_real_target_xform.inverse() * needletip_wrt_drill * drill_xform;
  
  FrameTransformList drilltarget_wrt_cam = {drill_safe_target_wrt_cam, drill_real_target_wrt_cam};
  
  return drilltarget_wrt_cam;
}


int main(int argc, char* argv[])
{
  #ifdef JHMR_HAS_OPENCL
    typedef CameraRayCasterGPULineIntegral RayCaster;
  #else
    typedef CameraRayCasterCPULineIntegral<float> RayCaster;
  #endif
  typedef RayCaster::CameraModelType CamModel;
  typedef RayCaster::ImageVolumeType ImageVolumeType;
  
  ProgOpts po;

  jhmrPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Compute 3D Polaris Position of Snake Tip Jig");
  po.set_arg_usage("<root regi debug H5 file source path> <root slicer H5 file source path> <exp ID file>");
  po.set_min_num_pos_args(3);

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
  const std::string current_URkin_h5            = po.pos_args()[0]; // Current UR kinematics h5 file as input
  const std::string target_URkin_path           = po.pos_args()[1]; // Target UR kinematics h5 file as output
  const std::string target_drill_path           = po.pos_args()[2]; // Target drill xform wrt Carm
  const std::string drillref_fcsv_path          = po.pos_args()[3]; // 3D drill HandeyeRef landmark path
  const std::string handeyeX_h5                 = po.pos_args()[4]; // HandeyeX matrix from pnp calibration
  const std::string spine_xform_h5              = po.pos_args()[5]; // Spine registration xform
  const std::string drill_xform_h5              = po.pos_args()[6]; // Drill registration xform
  const std::string needle_landmark_fcsv_path   = po.pos_args()[7]; // Needle landmark fcsv
  const std::string planning_landmark_fcsv_path = po.pos_args()[8]; // Planning landmark fcsv
  const std::string ID_prefix                   = po.pos_args()[9]; // Exp ID prefix, for example: 1
  
  std::cout << "reading drill rotation center ref landmark from FCSV file..." << std::endl;
  auto drillref_3dfcsv = ReadFCSVFileNamePtMap<Pt3>(drillref_fcsv_path);
  FromMapConvertRASToLPS(drillref_3dfcsv.begin(), drillref_3dfcsv.end());
  
  FrameTransform drill_handeye_ref = FrameTransform::Identity();
  {
    auto drillref_fcsv = drillref_3dfcsv.find("HandeyeRef");
    Pt3 drill_rotcen_pt;
    
    if (drillref_fcsv != drillref_3dfcsv.end()){
      drill_rotcen_pt = drillref_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND DRILL REF PT" << std::endl;
    }
    
    drill_handeye_ref.matrix()(0,3) = -drill_rotcen_pt[0];
    drill_handeye_ref.matrix()(1,3) = -drill_rotcen_pt[1];
    drill_handeye_ref.matrix()(2,3) = -drill_rotcen_pt[2];
  }
  
  const CamModel default_cam = NaiveCamModelFromCIOSFusion(
  MakeNaiveCIOSFusionMetaDR(), true).cast<CoordScalar>();
  
  // Read spine xform
  FrameTransform spine_xform;
  ReadITKAffineTransformFromFile(spine_xform_h5, &spine_xform);
  
  // Read drill xform
  FrameTransform drill_xform;
  ReadITKAffineTransformFromFile(drill_xform_h5, &drill_xform);
  
  FrameTransformList drilltarget_wrt_cam = Target_Drillpose_from_Planning(spine_xform, drill_xform, planning_landmark_fcsv_path, needle_landmark_fcsv_path, ID_prefix);
  
  FrameTransform drill_safe_target_wrt_cam = drilltarget_wrt_cam[0];
  FrameTransform drill_real_target_wrt_cam = drilltarget_wrt_cam[1];
  
  const std::string safe_target_drill_h5 = target_drill_path + "/safe_target_xform" + ID_prefix + ".h5";
  const std::string real_target_drill_h5 = target_drill_path + "/real_target_xform" + ID_prefix + ".h5";
  
  WriteITKAffineTransform(safe_target_drill_h5, drill_safe_target_wrt_cam);
  WriteITKAffineTransform(real_target_drill_h5, drill_real_target_wrt_cam);
  
  // Read Robot End Effector transformation from h5_slicer file
  H5::H5File h5_ureef(current_URkin_h5, H5F_ACC_RDWR);
  H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
  H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
  std::vector<float> UReef_tracker           = ReadVectorH5<float>("TranformParameters", ureef_group0);
  
  FrameTransform current_URkinematics        = ConvertSlicerToITK(UReef_tracker);
    
  FrameTransform handeyeX_xform;
  ReadITKAffineTransformFromFile(handeyeX_h5, &handeyeX_xform);
 
  FrameTransform safe_target_URkinematics = current_URkinematics * handeyeX_xform * drill_handeye_ref * drill_xform * drill_safe_target_wrt_cam.inverse() * drill_handeye_ref.inverse() * handeyeX_xform.inverse();
  FrameTransform real_target_URkinematics = current_URkinematics * handeyeX_xform * drill_handeye_ref * drill_xform * drill_real_target_wrt_cam.inverse() * drill_handeye_ref.inverse() * handeyeX_xform.inverse();
 
  const std::string safe_target_URkin_h5 = target_URkin_path + "/safe_target_ur" + ID_prefix + ".h5";
  const std::string real_target_URkin_h5 = target_URkin_path + "/real_target_ur" + ID_prefix + ".h5";
  
  FrameTransform safe_target_URkinematics_slicer = ConvertITKToSlicer(safe_target_URkinematics);
  FrameTransform real_target_URkinematics_slicer = ConvertITKToSlicer(real_target_URkinematics);
  
  // Write target UR kinematics to file
  WriteITKAffineTransform(safe_target_URkin_h5, safe_target_URkinematics_slicer);
  WriteITKAffineTransform(real_target_URkin_h5, real_target_URkinematics_slicer);
  
  std::cout << "Current UR kinematics:" << std::endl;
  std::cout << current_URkinematics.matrix() << std::endl;
  
  std::cout << "Safe Target UR kinematics:" << std::endl;
  std::cout << safe_target_URkinematics.matrix() << std::endl;
  
  std::cout << "Real Target UR kinematics:" << std::endl;
  std::cout << real_target_URkinematics.matrix() << std::endl;
    
  return kEXIT_VAL_SUCCESS;
}


