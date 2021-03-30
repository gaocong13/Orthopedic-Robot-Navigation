// STD
#include <iostream>
#include <vector>

#include <math.h>

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

int main(int argc, char* argv[])
{
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
  const std::string spine_xform_path              = po.pos_args()[0]; // Spine registration xform path
  const std::string drill_xform_path              = po.pos_args()[1]; // Drill registration xform path
  const std::string needle_landmark_fcsv_path     = po.pos_args()[2]; // 3D drill needle tip & base landmarks path
  const std::string planning_landmark_fcsv_path   = po.pos_args()[3]; // 3D planning entry & target points path
  const std::string drilltarget_xform_path        = po.pos_args()[4]; // Target Drill Pose path
  
  // Read spine xform
  FrameTransform spine_xform;
  ReadITKAffineTransformFromFile(spine_xform_path, &spine_xform);
  
  // Read drill xform
  FrameTransform drill_xform;
  ReadITKAffineTransformFromFile(drill_xform_path, &drill_xform);
  
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
  
  Pt3 target_pt;
  Pt3 entry_pt;
  {
    auto entry_fcsv = planning_landmark_3dfcsv.find("Entry");
    
    if (entry_fcsv != planning_landmark_3dfcsv.end()){
      entry_pt = entry_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND ENTRY PT" << std::endl;
    }
    
    auto target_fcsv = planning_landmark_3dfcsv.find("Target");
    
    if (target_fcsv != planning_landmark_3dfcsv.end()){
      target_pt = target_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND TARGET PT" << std::endl;
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
  Pt3 NT_entry_pt = needletip_ref_wrt_spine * entry_pt;
  
  // Transform target point from spine frame to needletip (NT) frame
  Pt3 NT_target_pt = needletip_ref_wrt_spine * target_pt;
  
  // Calculate vector from tip to base in needletip frame
  Pt3 NT_vec_base_to_tip = NT_needlebase_pt - NT_needletip_pt;
  
  // Calculate vector from entry to target point in needletip frame
  Pt3 NT_vec_entry_to_target = NT_entry_pt - NT_target_pt;
  
  // Construct cisst vectors
  vct3 vct3_base_to_tip(NT_vec_base_to_tip[0], NT_vec_base_to_tip[1], NT_vec_base_to_tip[2]);
  vct3 vct3_entry_to_target(NT_vec_entry_to_target[0], NT_vec_entry_to_target[1], NT_vec_entry_to_target[2]);
  
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
  
  // Compute final transformation
  FrameTransform drilltarget_wrt_carm = needletip_wrt_drill.inverse() * needletip_wrt_target_xform.inverse() * needletip_wrt_drill * drill_xform;
  
  WriteITKAffineTransform(drilltarget_xform_path, drilltarget_wrt_carm);
  return kEXIT_VAL_SUCCESS;
}


