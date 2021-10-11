// STD
#include <iostream>
#include <vector>

#include <fmt/format.h>

#include "xregProgOptUtils.h"
#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregITKLabelUtils.h"
#include "xregLandmarkMapUtils.h"
#include "xregAnatCoordFrames.h"
#include "xregH5ProjDataIO.h"
#include "xregRayCastProgOpts.h"
#include "xregRayCastInterface.h"
#include "xregImgSimMetric2DPatchCommon.h"
#include "xregImgSimMetric2DGradImgParamInterface.h"
#include "xregImgSimMetric2DProgOpts.h"
#include "xregHUToLinAtt.h"
#include "xregProjPreProc.h"
#include "xregCIOSFusionDICOM.h"
#include "xregPnPUtils.h"
#include "xregMultiObjMultiLevel2D3DRegi.h"
#include "xregMultiObjMultiLevel2D3DRegiDebug.h"
#include "xregSE3OptVars.h"
#include "xregIntensity2D3DRegiExhaustive.h"
#include "xregIntensity2D3DRegiCMAES.h"
#include "xregIntensity2D3DRegiBOBYQA.h"
#include "xregRegi2D3DPenaltyFnSE3Mag.h"
#include "xregFoldNormDist.h"
#include "xregHipSegUtils.h"
#include "xregHDF5.h"

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

  po.set_help("Compute 3D Polaris Position of Snake Tip Jig");
  po.set_arg_usage("<Source Bayview H5 root path> <Bayview H5 Debug file root path> <Bayview slicer root path> <Experiment list file path> <Result file path> <Top landmark path> <result path>");
  po.set_min_num_pos_args(4);

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

  const std::string meta_data_path             = po.pos_args()[0];  // Meta Data path
  const std::string dicom_path                 = po.pos_args()[1];  // Dicom image path
  const std::string init_xform_folder          = po.pos_args()[2];
  const std::string output_path                = po.pos_args()[3];
  const std::string device_exp_list_path       = po.pos_args()[4];  // Experiment image list file path
  const std::string spine_exp_list_path        = po.pos_args()[5];  // Experiment image list file path

  using UseCurEstForInit = MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst;

  const std::string spinevol_path = meta_data_path + "/Spine21-2512_CT_crop.nrrd";
  const std::string spineseg_path = meta_data_path + "/Spine21-2512_seg_crop.nrrd";
  const std::string sacrumseg_path = meta_data_path + "/Spine21-2512_sacrum_seg_crop.nrrd";
  const std::string spine_3d_fcsv_path = meta_data_path + "/Spine_3D_landmarks.fcsv";

  std::cout << "reading spine anatomical landmarks from FCSV file..." << std::endl;
  std::cout << spine_3d_fcsv_path << std::endl;
  auto spine_3d_fcsv = ReadFCSVFileNamePtMap(spine_3d_fcsv_path);
  ConvertRASToLPS(&spine_3d_fcsv);

  const bool use_seg = true;
  auto spine_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(spineseg_path);
  
  auto sacrum_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(sacrumseg_path);

  vout << "reading spine volume..." << std::endl; // We only use the needle metal part
  auto spinevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(spinevol_path);
  
  {
    spinevol_hu->SetOrigin(spine_seg->GetOrigin());
    spinevol_hu->SetSpacing(spine_seg->GetSpacing());
  }

  vout << "  HU --> Att. ..." << std::endl;
  auto spinevol_att = HUToLinAtt(spinevol_hu.GetPointer());
  
  {
    sacrum_seg->SetOrigin(spinevol_att->GetOrigin());
    sacrum_seg->SetSpacing(spinevol_att->GetSpacing());
  }
  
  const unsigned char vert1_label = 25;
  const unsigned char vert2_label = 24;
  const unsigned char vert3_label = 23;
  const unsigned char vert4_label = 22;
  const unsigned char sacrum_label = 1;

  unsigned char spine_label = 1;
  vout << "extracting vert label volumes..." << std::endl;
  auto vert1_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert1_label, float(0), true);
  
  auto vert2_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert2_label, float(0), true);
  
  auto vert3_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert3_label, float(0), true);
  
  auto vert4_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert4_label, float(0), true);
  
  auto sacrum_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), sacrum_seg.GetPointer(), sacrum_label, float(0), true);

  std::vector<std::string> spine_exp_ID_list;
  size_type spine_lineNumber = 0;
  /* Read exp ID list from file */
  {
    std::ifstream expIDFile(spine_exp_list_path);
    // Make sure the file is open
    if(!expIDFile.is_open()) throw std::runtime_error("Could not open Spine exp ID file");

    std::string line, csvItem;

    while(std::getline(expIDFile, line)){
      std::istringstream myline(line);
      while(getline(myline, csvItem)){
          spine_exp_ID_list.push_back(csvItem);
      }
      spine_lineNumber++;
    }
  }

  if(spine_lineNumber!=spine_exp_ID_list.size()) throw std::runtime_error("Spine Exp ID list size mismatch!!!");

  std::vector<std::string> device_exp_ID_list;
  size_type device_lineNumber = 0;
  /* Read exp ID list from file */
  {
    std::ifstream expIDFile(device_exp_list_path);
    // Make sure the file is open
    if(!expIDFile.is_open()) throw std::runtime_error("Could not open Device exp ID file");

    std::string line, csvItem;

    while(std::getline(expIDFile, line)){
      std::istringstream myline(line);
      while(getline(myline, csvItem)){
          device_exp_ID_list.push_back(csvItem);
      }
      device_lineNumber++;
    }
  }

  if(device_lineNumber!=device_exp_ID_list.size()) throw std::runtime_error("Device Exp ID list size mismatch!!!");

  const size_type num_views = spine_lineNumber;

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> spine_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> spine_multiview_regi_ref_frame;
  {
    vout << "setting up device ref. frame..." << std::endl;
    // setup camera aligned reference frame, use metal device volume center point as the origin

    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("RotCenter");
    Pt3 spine_rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      spine_rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND device ROT CENTER" << std::endl;
    }

    spine_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    spine_singleview_regi_ref_frame->vol_idx = 0;
    spine_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = spine_rotcenter[0];
    spine_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = spine_rotcenter[1];
    spine_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = spine_rotcenter[2];

    FrameTransform spine_vol_to_centered_vol = FrameTransform::Identity();
    spine_vol_to_centered_vol.matrix()(0,3) = -spine_rotcenter[0];
    spine_vol_to_centered_vol.matrix()(1,3) = -spine_rotcenter[1];
    spine_vol_to_centered_vol.matrix()(2,3) = -spine_rotcenter[2];

    spine_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(spine_vol_to_centered_vol, true);
  }
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert1_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> vert1_multiview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert1-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND vert1 center" << std::endl;
    }

    vert1_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    vert1_singleview_regi_ref_frame->vol_idx = 0;
    vert1_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    vert1_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    vert1_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
    
    FrameTransform vertebra_vol_to_centered_vol = FrameTransform::Identity();
    vertebra_vol_to_centered_vol.matrix()(0,3) = -rotcenter[0];
    vertebra_vol_to_centered_vol.matrix()(1,3) = -rotcenter[1];
    vertebra_vol_to_centered_vol.matrix()(2,3) = -rotcenter[2];

    vert1_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(vertebra_vol_to_centered_vol, true);
  }
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert2_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> vert2_multiview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert2-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND vert2 center" << std::endl;
    }

    vert2_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    vert2_singleview_regi_ref_frame->vol_idx = 1;
    vert2_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    vert2_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    vert2_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
    
    FrameTransform vertebra_vol_to_centered_vol = FrameTransform::Identity();
    vertebra_vol_to_centered_vol.matrix()(0,3) = -rotcenter[0];
    vertebra_vol_to_centered_vol.matrix()(1,3) = -rotcenter[1];
    vertebra_vol_to_centered_vol.matrix()(2,3) = -rotcenter[2];

    vert2_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(vertebra_vol_to_centered_vol, true);
  }
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert3_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> vert3_multiview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert3-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND vert3 center" << std::endl;
    }

    vert3_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    vert3_singleview_regi_ref_frame->vol_idx = 2;
    vert3_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    vert3_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    vert3_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
    
    FrameTransform vertebra_vol_to_centered_vol = FrameTransform::Identity();
    vertebra_vol_to_centered_vol.matrix()(0,3) = -rotcenter[0];
    vertebra_vol_to_centered_vol.matrix()(1,3) = -rotcenter[1];
    vertebra_vol_to_centered_vol.matrix()(2,3) = -rotcenter[2];

    vert3_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(vertebra_vol_to_centered_vol, true);
  }
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert4_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> vert4_multiview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert4-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND vert4 center" << std::endl;
    }

    vert4_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    vert4_singleview_regi_ref_frame->vol_idx = 3;
    vert4_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    vert4_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    vert4_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
    
    FrameTransform vertebra_vol_to_centered_vol = FrameTransform::Identity();
    vertebra_vol_to_centered_vol.matrix()(0,3) = -rotcenter[0];
    vertebra_vol_to_centered_vol.matrix()(1,3) = -rotcenter[1];
    vertebra_vol_to_centered_vol.matrix()(2,3) = -rotcenter[2];

    vert4_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(vertebra_vol_to_centered_vol, true);
  }
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> sacrum_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> sacrum_multiview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("sacrum-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND sacrum center" << std::endl;
    }

    sacrum_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    sacrum_singleview_regi_ref_frame->vol_idx = 4;
    sacrum_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    sacrum_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    sacrum_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
    
    FrameTransform vertebra_vol_to_centered_vol = FrameTransform::Identity();
    vertebra_vol_to_centered_vol.matrix()(0,3) = -rotcenter[0];
    vertebra_vol_to_centered_vol.matrix()(1,3) = -rotcenter[1];
    vertebra_vol_to_centered_vol.matrix()(2,3) = -rotcenter[2];

    sacrum_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(vertebra_vol_to_centered_vol, true);
  }

  ProjPreProc proj_pre_proc;
  proj_pre_proc.input_projs.resize(num_views);
  std::vector<CIOSFusionDICOMInfo> spine_cios_metas(num_views);
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    std::string img_path_ID = dicom_path + "/" + spine_exp_ID_list[view_idx];
    {
      std::tie(proj_pre_proc.input_projs[view_idx].img, spine_cios_metas[view_idx]) = ReadCIOSFusionDICOMFloat(img_path_ID);
      proj_pre_proc.input_projs[view_idx].cam = NaiveCamModelFromCIOSFusion(spine_cios_metas[view_idx], true);
    }
  }

  proj_pre_proc();

  auto& projs_to_regi = proj_pre_proc.output_projs;

  FrameTransformList init_device_xforms;

  init_device_xforms.reserve(num_views);

  for(size_type view_idx = 0;view_idx < num_views; ++view_idx)
  {
    const std::string device_xform_file_path = init_xform_folder + "/device_regi_xform" + device_exp_ID_list[view_idx] + ".h5";
    init_device_xforms[view_idx] = ReadITKAffineTransformFromFile(device_xform_file_path);
  }

  std::string vert1_xform_file_path = init_xform_folder + "/vert1_regi_xform.h5";
  std::string vert2_xform_file_path = init_xform_folder + "/vert2_regi_xform.h5";
  std::string vert3_xform_file_path = init_xform_folder + "/vert3_regi_xform.h5";
  std::string vert4_xform_file_path = init_xform_folder + "/vert4_regi_xform.h5";
  std::string sacrum_xform_file_path = init_xform_folder + "/sacrum_regi_xform.h5";
  
  std::vector<std::string> vertebras_init_xform_path_list = {vert1_xform_file_path, vert2_xform_file_path, vert3_xform_file_path, vert4_xform_file_path, sacrum_xform_file_path};
  
  FrameTransformList init_spine_xform;
  
  for(size_type vert_idx=0; vert_idx<5; ++vert_idx)
  {
    FrameTransform temp_vert_xform = ReadITKAffineTransformFromFile(vertebras_init_xform_path_list[vert_idx]);
    init_spine_xform.push_back(temp_vert_xform);
  }

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  std::vector<CameraModel> orig_cams;
  for (auto& pd : projs_to_regi)
  {
    orig_cams.push_back(pd.cam);
  }

  // Using device regi as fiducial
  auto cams_devicefid = CreateCameraWorldUsingFiducial(orig_cams, init_device_xforms);

  FrameTransform init_device_cam_to_vols = init_device_xforms[0];

  MultiLevelMultiObjRegi regi;

  regi.set_debug_output_stream(vout, verbose);
  regi.set_save_debug_info(kSAVE_REGI_DEBUG);
  regi.vols = { vert1_vol, vert2_vol, vert3_vol, vert4_vol, sacrum_vol };
  regi.vol_names = { "vert1", "vert2", "vert3", "vert4", "sacrum" };

  regi.ref_frames = { vert1_multiview_regi_ref_frame, vert2_multiview_regi_ref_frame,
    vert3_multiview_regi_ref_frame, vert4_multiview_regi_ref_frame, sacrum_multiview_regi_ref_frame };

  regi.fixed_proj_data = proj_pre_proc.output_projs;

  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    regi.fixed_proj_data[view_idx].cam = cams_devicefid[view_idx];
  }

  regi.levels.resize(1);
  
  FrameTransformList init_spine_xform_devicefid;
  for(size_type vert_idx=0; vert_idx<5; ++vert_idx)
  {
    init_spine_xform_devicefid.push_back( init_spine_xform[vert_idx] * init_device_cam_to_vols.inverse());
  }

  regi.init_cam_to_vols = { init_spine_xform_devicefid[0], init_spine_xform_devicefid[1],
    init_spine_xform_devicefid[2], init_spine_xform_devicefid[3], init_spine_xform_devicefid[4] };

  auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();

  // Multi-view Device Registration
  {
    auto& lvl = regi.levels[0];

    lvl.ds_factor = 0.25;

    lvl.fixed_imgs_to_use.resize(num_views);
    std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

    vout << "    setting up ray caster..." << std::endl;
    lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

    vout << "    setting up sim metrics..." << std::endl;
    lvl.sim_metrics.reserve(num_views);

    for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
    {
      auto sm = PatchGradNCCSimMetricFromProgOpts(po);

      {
        auto* grad_sm = dynamic_cast<ImgSimMetric2DGradImgParamInterface*>(sm.get());

        grad_sm->set_smooth_img_before_sobel_kernel_radius(5);
      }

      {
        auto* patch_sm = dynamic_cast<ImgSimMetric2DPatchCommon*>(sm.get());
        xregASSERT(patch_sm);

        patch_sm->set_patch_radius(std::lround(lvl.ds_factor * 41));
        patch_sm->set_patch_stride(1);
      }

      lvl.sim_metrics.push_back(sm);
    }


    lvl.regis.resize(1);

    {
      vout << "Setting up multiple-view spine regi..." << std::endl;

      auto& lvl_regi = lvl.regis[0];

      lvl_regi.mov_vols    = { 0, 1, 2, 3, 4 }; // This refers to device (moving)
      lvl_regi.static_vols = {  };

      auto vert1_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      vert1_guess->vol_idx = 0;
      
      auto vert2_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      vert2_guess->vol_idx = 1;
      
      auto vert3_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      vert3_guess->vol_idx = 2;
      
      auto vert4_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      vert4_guess->vol_idx = 3;
      
      auto sacrum_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      sacrum_guess->vol_idx = 4;

      lvl_regi.ref_frames = { 0, 1, 2, 3, 4 };

      lvl_regi.init_mov_vol_poses = { vert1_guess, vert2_guess, vert3_guess, vert4_guess, sacrum_guess };

      // Set CMAES parameters
      auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(0.01);
      cmaes_regi->set_opt_obj_fn_tol(0.01);

      auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

      pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD),
                                     std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD),
                                     std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD),
                                     std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD),
                                     std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };

      pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(20, 20),
                                     std::make_shared<FoldNormDist>(20, 20),
                                     std::make_shared<FoldNormDist>(20, 20),
                                     std::make_shared<FoldNormDist>(20, 20),
                                     std::make_shared<FoldNormDist>(20, 20) };

      cmaes_regi->set_pop_size(50);
      cmaes_regi->set_sigma({ 1 * kDEG2RAD, 1 * kDEG2RAD, 1 * kDEG2RAD, 2, 2, 5,
                              1 * kDEG2RAD, 1 * kDEG2RAD, 1 * kDEG2RAD, 2, 2, 5,
                              1 * kDEG2RAD, 1 * kDEG2RAD, 1 * kDEG2RAD, 2, 2, 5,
                              1 * kDEG2RAD, 1 * kDEG2RAD, 1 * kDEG2RAD, 2, 2, 5,
                              1 * kDEG2RAD, 1 * kDEG2RAD, 1 * kDEG2RAD, 2, 2, 5 });

      cmaes_regi->set_penalty_fn(pen_fn);
      cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

      lvl_regi.regi = cmaes_regi;
    }
  }

  if (kSAVE_REGI_DEBUG )
  {
    vout << "  setting regi debug info..." << std::endl;

    // Create Debug Proj H5 File
    std::string joint_expID = "";
    for(size_type view_idx = 0;view_idx < num_views; ++view_idx)
    {
      joint_expID += "_" + spine_exp_ID_list[view_idx];
    }
    const std::string proj_data_h5_path = output_path + "/proj_data" + joint_expID + ".h5";
    vout << "creating H5 proj data file ..." << std::endl;
    H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
    WriteProjDataH5(regi.fixed_proj_data, &h5);
    h5.flush(H5F_SCOPE_GLOBAL);
    h5.close();

    DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
    debug_vol_path.vol_path = spinevol_path;

    if (use_seg)
    {
      debug_vol_path.label_vol_path = spineseg_path;
      debug_vol_path.labels_used    = { spine_label };
    }

    regi.debug_info->vols = { debug_vol_path };

    DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
    debug_proj_path.path = proj_data_h5_path;
    // debug_proj_path.projs_used = { view_idx };

    regi.debug_info->fixed_projs = debug_proj_path;

    regi.debug_info->proj_pre_proc_info = proj_pre_proc.params;

    regi.debug_info->regi_names = { { "Multiview Spine Regi" } };
  }

  vout << std::endl << " ************  Running Multi-view Spine Registration ... " << std::endl;
  regi.run();

  if (kSAVE_REGI_DEBUG)
  {
    vout << "writing debug info to disk..." << std::endl;
    const std::string dst_debug_path = output_path + "/debug_vertb_mv.h5";
    WriteMultiLevel2D3DRegiDebugToDisk(*regi.debug_info, dst_debug_path);
  }

  FrameTransform spine_regi_xform = regi.cur_cam_to_vols[0];

  vout << "saving transformations..." << std::endl;
  FrameTransform vert1_regi_xform = regi.cur_cam_to_vols[0];
  FrameTransform vert2_regi_xform = regi.cur_cam_to_vols[1];
  FrameTransform vert3_regi_xform = regi.cur_cam_to_vols[2];
  FrameTransform vert4_regi_xform = regi.cur_cam_to_vols[3];
  FrameTransform sacrum_regi_xform = regi.cur_cam_to_vols[4];
  
  WriteITKAffineTransform(output_path + "/vert1_regi_xform.h5", vert1_regi_xform);
  WriteITKAffineTransform(output_path + "/vert2_regi_xform.h5", vert2_regi_xform);
  WriteITKAffineTransform(output_path + "/vert3_regi_xform.h5", vert3_regi_xform);
  WriteITKAffineTransform(output_path + "/vert4_regi_xform.h5", vert4_regi_xform);
  WriteITKAffineTransform(output_path + "/sacrum_regi_xform.h5", sacrum_regi_xform);

  {
    for(size_type view_idx=0; view_idx<num_views; ++view_idx)
    {
      for(size_type vol_idx=0; vol_idx<5; ++vol_idx)
      {
        auto ray_caster = LineIntRayCasterFromProgOpts(po);
        ray_caster->set_camera_model(regi.fixed_proj_data[view_idx].cam);
        ray_caster->use_proj_store_replace_method();
        ray_caster->set_volume(regi.vols[vol_idx]);
        ray_caster->set_num_projs(1);
        ray_caster->allocate_resources();
        ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[vol_idx];
        ray_caster->compute(0);

        WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/vert_reproj_vert" + fmt::format("{:02d}", vol_idx) + "_view" + fmt::format("{:02d}", view_idx) + ".png");
      }
    }
    WriteITKImageRemap8bpp(proj_pre_proc.input_projs[0].img.GetPointer(), output_path + "/real.png");
  }
  
  return 0;
}
