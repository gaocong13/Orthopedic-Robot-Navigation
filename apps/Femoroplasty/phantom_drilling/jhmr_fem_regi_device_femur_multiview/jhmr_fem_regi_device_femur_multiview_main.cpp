
// STD
#include <iostream>
#include <vector>

// Boost
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/null.hpp>

#include <fmt/format.h>

// jhmr
#include "jhmrProgOptUtils.h"
#include "jhmrPAOUtils.h"
//#include "jhmrTimer.h"
//#include "jhmrProjDataIO.h"
#include "jhmrHDF5.h"
#include "jhmrRegi2D3DDebugH5.h"
#include "jhmrRegi2D3DDebugIO.h"
#include "jhmrMultiObjMultiLevel2D3DRegi.h"
#include "jhmrFCSVUtils.h"
#include "jhmrProjPreProc.h"
#include "jhmrCIOSFusionDICOM.h"
#include "jhmrHounsfieldToLinearAttenuationFilter.h"
#include "jhmrGPUPrefsXML.h"
#include "jhmrRegi2D3DCMAES.h"
#include "jhmrRegi2D3DBOBYQA.h"
#include "jhmrPatchGradNCCImageSimilarityMetricGPU.h"
#include "jhmrNormalDist.h"
#include "jhmrFoldedNormalDist.h"
#include "jhmrRegi2D3DSE3EulerDecompPenaltyFn.h"
#include "jhmrRegi2D3DSE3MagPenaltyFn.h"
#include "jhmrCameraRayCastingGPU.h"
#include "jhmrAnatCoordFrames.h"
#include "jhmrLandmark2D3DRegi.h"
#include "jhmrRegi2D3DIntensityExhaustive.h"
#include "jhmrSampleUniformUnitVecs.h"

#include "CommonFEM.h"

using namespace jhmr;
using namespace jhmr::fem;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;

using Cam                 = MultiLevelMultiObjRegi::CamImgPair::CamModel;
using Proj                = MultiLevelMultiObjRegi::CamImgPair::ImageType;
using RayCasterLineIntGPU = CameraRayCasterGPULineIntegral;
using CMAESRegi           = Regi2D3DCMAES<MultiLevelMultiObjRegi::RayCaster,
MultiLevelMultiObjRegi::SimMetric>;
using BOBYQARegi     = Regi2D3DBOBYQA<MultiLevelMultiObjRegi::RayCaster,
MultiLevelMultiObjRegi::SimMetric>;

using SimMetGPU = PatchGradNCCImageSimilarityMetricGPU;
using SimMetCPU = PatchGradNCCImageSimilarityMetricCPU<PixelScalar>;

using ExhaustiveRegi      = Regi2D3DExhaustive<MultiLevelMultiObjRegi::RayCaster,
MultiLevelMultiObjRegi::SimMetric>;

using FoldNormPDF = FoldedNormalPDF<CoordScalar>;
using RotMagAndTransMagPen = Regi2D3DSE3MagPenaltyFn<FoldNormPDF>;

typedef RayCasterLineIntGPU::ImageVolumeType ImageVolumeType;

auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();
auto so3_vars = std::make_shared<SO3OptVarsLieAlg<double>>();

//constexpr bool kSAVE_REGI_DEBUG = true;
const bool debug_multi_view = false;

namespace{
// used to apply no regularization to the translation component
// when we are optimizing, and only changing, the rotation component
struct NullDist
{
  using Scalar = CoordScalar;

  static constexpr Scalar norm_const = 1;

  static constexpr Scalar log_norm_const = 0;

  Scalar operator()(const Scalar x) const
  {
    return 1;
  }

  Scalar log_density(const Scalar x) const
  {
    return 0;
  }
};

} // un-named

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
  #ifdef JHMR_HAS_OPENCL
    typedef CameraRayCasterGPULineIntegral RayCaster;
  #else
    typedef CameraRayCasterCPULineIntegral<float> RayCaster;
  #endif
  
  // Set up the program options
  ProgOpts po;

  jhmrPROG_OPTS_SET_COMPILE_DATE(po);
  // allow negative numbers (e.g. leading with -) positional arguments
  //po.set_allow_unrecognized_flags(true);

  po.set_help("Example driver for a multi-view femur registration using femur as fiducial object.");
  po.set_arg_usage("<src hdf5 path> <CT intensity 3D volume> <Drill 3D volume> <3D volume segmentation> "
                   "<Drill Landmarks FCSV> <Output Path>");
  po.set_min_num_pos_args(6);

  po.add("verbose", 'v', ProgOpts::kSTORE_TRUE, "verbose", "Verbose logging to stdout")
    << false;  // default to non-verbose mode
  po.add("debug", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "debug", "Save debug info")
    << false;  // default to not save
  po.add("init-device-with-offset", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "init-device-with-offset", "Initialize Drill Transformation with a delta offset")
  << false;  // default to not save

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

  boost::iostreams::stream<boost::iostreams::null_sink> null_ostream((boost::iostreams::null_sink()));
  const bool verbose = po.get("verbose");
  std::ostream& vout = verbose ? std::cout : null_ostream;
  const bool kSAVE_REGI_DEBUG = po.get("debug");
  const bool init_device_with_offset = po.get("init-device-with-offset");

  const std::string device_3d_fcsv_path        = po.pos_args()[0];
  const std::string devicebb_2d_fcsv_path      = po.pos_args()[1];  // 2D device landmarks path
  const std::string femurld_2d_fcsv_path       = po.pos_args()[2];  // 2D femur landmarks path
  const std::string init_xform_folder          = po.pos_args()[3];
  const std::string output_path                = po.pos_args()[4];
  const std::string img_path                   = po.pos_args()[5];
  
  std::vector<std::string> img_ID_list(po.pos_args().begin()+6, po.pos_args().end()); // First 3 are soft tissue; Last 3 are with device
  
  const std::string femur_vol_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/meta_data/Femur_CT_crop.nii.gz";
  const std::string femur_seg_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/meta_data/Femur_seg_crop.nii.gz";
  const std::string device_vol_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/meta_data/Device_cropmore_CT.nii.gz";
  const std::string device_seg_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/meta_data/Device_cropmore_seg.nii.gz";
  
  const std::string femurld_3d_fcsv_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/meta_data/Femur3Dlandmarks.fcsv";
  const std::string devicebb_3d_fcsv_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/meta_data/Device3Dlandmark.fcsv";
  
  const size_type num_views = 3;//Hard code
  const LabelScalar femur_label = 1;
  const LabelScalar device_label = 1;
  
  const CamModel default_cam = NaiveCamModelFromCIOSFusion(
  MakeNaiveCIOSFusionMetaDR(), true).cast<CoordScalar>();

  vout << "default GPU prefs..." << std::endl;
  GPUPrefsXML gpu_prefs;
  
  VolPtr femur_vol_att;
  {
    vout << "reading input CT volume..." << std::endl;
    auto vol_hu = ReadITKImageFromDisk<Vol>(femur_vol_path);

    vout << "  HU --> Att. ..." << std::endl;
    auto hu2att = HounsfieldToLinearAttenuationFilter<Vol>::New();
    hu2att->SetInput(vol_hu);
    hu2att->Update();
    
    femur_vol_att = hu2att->GetOutput();
  }

  LabelVolPtr device_seg = ReadITKImageFromDisk<LabelVol>(device_seg_path);
  
  VolPtr devicevol_att;
  {
    vout << "reading device volume..." << std::endl; // We only use the needle metal part
    auto devicevol_hu = ReadITKImageFromDisk<Vol>(device_vol_path);

    vout << "  HU --> Att. ..." << std::endl;
    auto hu2att = HounsfieldToLinearAttenuationFilter<Vol>::New();
    hu2att->SetInput(devicevol_hu);
    hu2att->Update();
    
    devicevol_att = hu2att->GetOutput();
  }
  
  VolPtr device_vol = ApplyMaskToITKImage(devicevol_att.GetPointer(), device_seg.GetPointer(), device_label, PixelScalar(0), true);

  vout << "reading volume segmentation..." << std::endl;
  LabelVolPtr femur_seg = ReadITKImageFromDisk<LabelVol>(femur_seg_path);

  vout << "extracting femur att. volume..." << std::endl;
  VolPtr femur_vol = ApplyMaskToITKImage(femur_vol_att.GetPointer(), femur_seg.GetPointer(), femur_label, PixelScalar(0), true);

  vout << "reading device tip from FCSV file..." << std::endl;
  auto device_fcsv = ReadFCSVFileNamePtMap<Pt3>(device_3d_fcsv_path);
  vout << "  RAS --> LPS..." << std::endl;
  FromMapConvertRASToLPS(device_fcsv.begin(), device_fcsv.end());
  
  vout << "reading device 2d BBs from FCSV file..." << std::endl;
  auto devicebb_2d_fcsv = ReadFCSVFileNamePtMap<Pt3>(devicebb_2d_fcsv_path);
  FromMapConvertRASToLPS(devicebb_2d_fcsv.begin(), devicebb_2d_fcsv.end());
  
  vout << "reading femur 2d landmarks from FCSV file..." << std::endl;
  auto femurld_2d_fcsv = ReadFCSVFileNamePtMap<Pt3>(femurld_2d_fcsv_path);
  FromMapConvertRASToLPS(femurld_2d_fcsv.begin(), femurld_2d_fcsv.end());
  
  vout << "reading device 3d BBs from FCSV file..." << std::endl;
  auto devicebb_3d_fcsv = ReadFCSVFileNamePtMap<Pt3>(devicebb_3d_fcsv_path);
  FromMapConvertRASToLPS(devicebb_3d_fcsv.begin(), devicebb_3d_fcsv.end());
  
  vout << "reading femur 3d Landmarks from FCSV file..." << std::endl;
  auto femurld_3d_fcsv = ReadFCSVFileNamePtMap<Pt3>(femurld_3d_fcsv_path);
  FromMapConvertRASToLPS(femurld_3d_fcsv.begin(), femurld_3d_fcsv.end());
  
  ProjPreProc<PixelScalar> proj_st_pre_proc;
  ProjPreProc<PixelScalar> proj_dr_pre_proc;
  proj_dr_pre_proc.input_projs.resize(num_views);
  proj_st_pre_proc.input_projs.resize(num_views);
  
  
  vout << "reading 2D softtissue images..." << std::endl;
  std::vector<CIOSFusionDICOMInfo> cios_metas_dr(num_views);
  
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    std::string img_path_ID = img_path + "/" + img_ID_list[view_idx];
    std::tie(proj_dr_pre_proc.input_projs[view_idx].img, cios_metas_dr[view_idx]) =
                                        ReadCIOSFusionDICOM<PixelScalar>(img_path_ID);
    proj_dr_pre_proc.input_projs[view_idx].cam = NaiveCamModelFromCIOSFusion(
                                        cios_metas_dr[view_idx], true).cast<CoordScalar>();
  }

  vout << "reading 2D device images..." << std::endl;
  std::vector<CIOSFusionDICOMInfo> cios_metas_st(num_views);
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    // DRILL: Read Simulation
    std::string img_path_ID = img_path + "/" + img_ID_list[view_idx+num_views];
    std::tie(proj_st_pre_proc.input_projs[view_idx].img, cios_metas_st[view_idx]) =
                                        ReadCIOSFusionDICOM<PixelScalar>(img_path_ID);
    proj_st_pre_proc.input_projs[view_idx].cam = NaiveCamModelFromCIOSFusion(
                                        cios_metas_st[view_idx], true).cast<CoordScalar>();
  }
  
  {
    UpdateLandmarkMapForCIOSFusion(cios_metas_st[0], femurld_2d_fcsv.begin(), femurld_2d_fcsv.end());
    auto& femurproj_lands = proj_st_pre_proc.input_projs[0].landmarks;
    femurproj_lands.reserve(femurld_2d_fcsv.size());

    for (const auto& fcsv_kv :femurld_2d_fcsv)
    {
      femurproj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
    }
  }
  
  {
    UpdateLandmarkMapForCIOSFusion(cios_metas_dr[0], devicebb_2d_fcsv.begin(), devicebb_2d_fcsv.end());
    auto& deviceproj_lands = proj_dr_pre_proc.input_projs[0].landmarks;
    deviceproj_lands.reserve(devicebb_2d_fcsv.size());

    for (const auto& fcsv_kv :devicebb_2d_fcsv)
    {
      deviceproj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
    }
  }

  vout << "running 2D preprocessing..." << std::endl;
  proj_st_pre_proc();
  proj_dr_pre_proc();

  auto& projs_st_to_regi = proj_st_pre_proc.output_projs;
  auto& projs_dr_to_regi = proj_dr_pre_proc.output_projs;

  vout << "Extract initialization from simulation..." << std::endl;
  
  FrameTransform gt_xform = EstCamToWorldBruteForcePOSITCMAESRefine(proj_st_pre_proc.output_projs[0].cam, proj_st_pre_proc.output_projs[0].landmarks, femurld_3d_fcsv);
  FrameTransform device_xform = EstCamToWorldBruteForcePOSITCMAESRefine(proj_dr_pre_proc.output_projs[0].cam, proj_dr_pre_proc.output_projs[0].landmarks, devicebb_3d_fcsv);
  //ReadITKAffineTransformFromFile(output_path + "/gt_xform.h5", &gt_xform);
  //ReadITKAffineTransformFromFile("/Users/gaocong/Documents/Research/Spine/Drill_Handeye/device_pnp/devicepnp_xformOct01-003.h5", &device_xform);
  
  FrameTransformList init_device_xform;
  FrameTransform init_femur_xform;
  
  init_device_xform.reserve(num_views);
  
  for(size_type view_idx = 0;view_idx < num_views; ++view_idx)
  {
    const std::string device_xform_file_path = init_xform_folder + "/drill_regi_xform" + img_ID_list[view_idx] + ".h5";
    ReadITKAffineTransformFromFile(device_xform_file_path, &init_device_xform[view_idx]);
  }
  
  const std::string femur_xform_file_path = init_xform_folder + "/femur_regi_xform" + img_ID_list[num_views] + ".h5";
  ReadITKAffineTransformFromFile(femur_xform_file_path, &init_femur_xform);

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> femur_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> femur_multiview_regi_ref_frame;
  {
    vout << "setting up femur ref. frame..." << std::endl;
    Pt3 femur_pt;

    auto check_label = [&femur_seg,&femur_pt,&femurld_3d_fcsv] (const std::string& k)
    {
      bool found = false;

      auto vol_fcsv_it = femurld_3d_fcsv.find(k);
    
      if (vol_fcsv_it != femurld_3d_fcsv.end())
      {
        femur_pt = vol_fcsv_it->second;

        itk::Point<CoordScalar,3> tmp_itk_pt;
        LabelVol::IndexType tmp_itk_idx;
        
        tmp_itk_pt[0] = femur_pt[0];
        tmp_itk_pt[1] = femur_pt[1];
        tmp_itk_pt[2] = femur_pt[2];
        
        femur_seg->TransformPhysicalPointToIndex(tmp_itk_pt, tmp_itk_idx);
        
        found = femur_seg->GetPixel(tmp_itk_idx) == femur_label;
      }
      
      return found;
    };

    bool found_femur_land = check_label("FH");
    
    if (found_femur_land)
    {
      vout << "  found left femoral head - will use as center of rotation for femur registration" << std::endl;
    }
  
    if (!found_femur_land)
    {
      jhmrThrow("ERROR: could not find appropriate femur landmark!!");
    }

    femur_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    femur_singleview_regi_ref_frame->vol_idx = 0;
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = femur_pt[0];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = femur_pt[1];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = femur_pt[2];
    
    FrameTransform femur_vol_to_centered_vol = FrameTransform::Identity();
    femur_vol_to_centered_vol.matrix()(0,3) = -femur_pt[0];
    femur_vol_to_centered_vol.matrix()(1,3) = -femur_pt[1];
    femur_vol_to_centered_vol.matrix()(2,3) = -femur_pt[2];

    femur_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(femur_vol_to_centered_vol, true);
  }
  
  auto exhaustive_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(FrameTransform::Identity(), false);

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> device_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> device_multiview_regi_ref_frame;
    
  {
    vout << "setting up device ref. frame..." << std::endl;
    // setup camera aligned reference frame, use metal device volume center point as the origin
  
    itk::ContinuousIndex<double,3> center_idx;

    auto device_fcsv_lsc = device_fcsv.find("RotCenter");
    Pt3 device_rotcenter;

    if (device_fcsv_lsc != device_fcsv.end()){
      device_rotcenter = device_fcsv_lsc->second;
    }
    else{
      vout << "ERROR: NOT FOUND DRILL ROT CENTER" << std::endl;
    }

    device_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    device_singleview_regi_ref_frame->vol_idx = 1;
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = device_rotcenter[0];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = device_rotcenter[1];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = device_rotcenter[2];

    FrameTransform device_vol_to_centered_vol = FrameTransform::Identity();
    device_vol_to_centered_vol.matrix()(0,3) = -device_rotcenter[0];
    device_vol_to_centered_vol.matrix()(1,3) = -device_rotcenter[1];
    device_vol_to_centered_vol.matrix()(2,3) = -device_rotcenter[2];

    device_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(device_vol_to_centered_vol, true);
  }
  
  Timer tmr;
  tmr.start();
  
  MultiLevelMultiObjRegi regi;

  regi.save_debug_info = kSAVE_REGI_DEBUG;
  regi.debug_info.vol_path       = device_vol_path;
  regi.debug_info.label_vol_path  = device_seg_path;
  regi.debug_info.labels_used = { device_label };
  regi.vols = { device_vol, femur_vol };
  regi.vol_names = { "Device", "Femur" };
    
  regi.ref_frames = { device_singleview_regi_ref_frame, device_multiview_regi_ref_frame, femur_multiview_regi_ref_frame };

  regi.fixed_cam_imgs.resize(num_views);
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    regi.fixed_cam_imgs[view_idx].img = projs_dr_to_regi[view_idx].img;
    regi.fixed_cam_imgs[view_idx].cam = projs_dr_to_regi[view_idx].cam;
  }
    
  regi.levels.resize(1);

  FrameTransformList regi_cams_to_device_vol(num_views);

  auto run_device_regi = [&gpu_prefs, &vout, &regi, &regi_cams_to_device_vol, &device_singleview_regi_ref_frame,
                         &init_device_xform, &init_femur_xform, &output_path, &kSAVE_REGI_DEBUG] (const size_type view_idx)
  {
    const bool is_first_view = view_idx == 0;

    regi.debug_info.regi_names = { { "Device-View" + fmt::sprintf("%03lu", view_idx) } };
    regi.init_cam_to_vols = { init_device_xform[view_idx], init_femur_xform };

    device_singleview_regi_ref_frame->cam_extrins = regi.fixed_cam_imgs[view_idx].cam.extrins;
    
    auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();
    
    auto& lvl = regi.levels[0];
    
    lvl.ds_factor = 0.25;
    
    lvl.fixed_imgs_to_use = { view_idx };

    auto rc = std::make_shared<RayCasterLineIntGPU>(gpu_prefs.ctx, gpu_prefs.queue);
    lvl.ray_caster = rc;
    {
      auto sm = std::make_shared<SimMetGPU>(gpu_prefs.ctx, gpu_prefs.queue);
      sm->set_setup_vienna_cl_ctx(is_first_view);
      sm->set_smooth_img_before_sobel_kernel_radius(5);
      sm->set_patch_radius(std::lround(lvl.ds_factor * 41));
      sm->set_patch_stride(5);
      lvl.sim_metrics = { sm };
    }
    
    lvl.regis.resize(1);

    {
      auto& lvl_regi_coarse = lvl.regis[0];
      lvl_regi_coarse.mov_vols = { 0 };
      lvl_regi_coarse.static_vols = { };

      auto init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      init_guess->vol_idx = 0;
      lvl_regi_coarse.init_mov_vol_poses = { init_guess };
      lvl_regi_coarse.ref_frames = { 0 };
      {
        auto cmaes_regi = std::make_shared<CMAESRegi>();
        cmaes_regi->set_opt_vars(se3_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        using NormPDF = NormalDist1D<CoordScalar>;

        auto pen_fn = std::make_shared<Regi2D3DSE3EulerDecompPenaltyFn<NormPDF>>();
      
        cmaes_regi->set_pop_size(100);
        cmaes_regi->set_sigma({ 5 * kDEG2RAD, 5 * kDEG2RAD, 5 * kDEG2RAD, 10, 10, 10 });
        
        pen_fn->rot_x_pdf   = NormPDF(0, 2.5 * kDEG2RAD);
        pen_fn->rot_y_pdf   = NormPDF(0, 2.5 * kDEG2RAD);
        pen_fn->rot_z_pdf   = NormPDF(0, 2.5 * kDEG2RAD);
        pen_fn->trans_x_pdf = NormPDF(0, 10);
        pen_fn->trans_y_pdf = NormPDF(0, 10);
        pen_fn->trans_z_pdf = NormPDF(0, 10);

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);
        
        lvl_regi_coarse.regi = cmaes_regi;
      }
      vout << std::endl << "View " << fmt::sprintf("%03lu", view_idx) << " device registration ..." << std::endl;
      regi.run();
      regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

      if(kSAVE_REGI_DEBUG){
        size_type img_num = regi.fixed_cam_imgs.size();
        std::vector<SingleProjData<PixelScalar>> tmp_pd_list(img_num);
        for (size_type v = 0; v < img_num; ++v)
        {
          tmp_pd_list[v].img = regi.fixed_cam_imgs[v].img;
          tmp_pd_list[v].cam = regi.fixed_cam_imgs[v].cam;
        }
        WriteRegi2D3DMultiLevelDebug(regi.debug_info, output_path+"/device_debug_view" + fmt::sprintf("%03lu", view_idx) + ".h5",
                                    std::make_tuple(VolPtr(), LabelVolPtr(), tmp_pd_list));
      }
      regi_cams_to_device_vol[view_idx] = regi.cur_cam_to_vols[0];
    }
  };

  for (size_type view_idx = 0; view_idx < num_views; ++view_idx){
    run_device_regi(view_idx);
  }

  std::vector<Cam> cams;
  for (auto& pd : projs_st_to_regi){
    cams.push_back(pd.cam);
  }
  std::vector<Cam> cams_devicefid = CreateCameraWorldUsingFiducial(cams, regi_cams_to_device_vol);

  for (size_type view_idx = 0; view_idx < num_views; ++view_idx){
    regi.fixed_cam_imgs[view_idx].cam = cams_devicefid[view_idx];
  }

  FrameTransform init_device_cam_to_vols = regi_cams_to_device_vol[0];
  
  // Femur Initialization
  FrameTransform init_femur_xform_devicefid = init_femur_xform * init_device_cam_to_vols.inverse();

  regi.init_cam_to_vols = { FrameTransform::Identity(), init_femur_xform_devicefid };
  regi.debug_info.regi_names = { { "Multiview Device" } };
  // Spine Registration
  {
    auto& lvl = regi.levels[0];
    lvl.ds_factor = 0.25;
    lvl.fixed_imgs_to_use.resize(num_views);
    std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

    auto rc = std::make_shared<RayCasterLineIntGPU>(gpu_prefs.ctx, gpu_prefs.queue);
    lvl.ray_caster = rc;
    {
      lvl.sim_metrics.resize(num_views);
      for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
      {
        auto sm = std::make_shared<PatchGradNCCImageSimilarityMetricGPU>(gpu_prefs.ctx, gpu_prefs.queue);
        sm->set_setup_vienna_cl_ctx(false);
        sm->set_smooth_img_before_sobel_kernel_radius(5);
        sm->set_patch_radius(std::lround(lvl.ds_factor * 41));
        sm->set_patch_stride(5);
        lvl.sim_metrics[view_idx] = sm;
      }
    }

    lvl.regis.resize(1);
    using FoldNormPDF = FoldedNormalPDF<CoordScalar>;
    using PenFn       = Regi2D3DSE3MagPenaltyFn<FoldNormPDF,NullDist>;
    {
      auto& lvl_regi = lvl.regis[0];;
      lvl_regi.mov_vols    = { 0 }; // This refers to device (moving)
      lvl_regi.static_vols = {  };
      auto device_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      device_init_guess->vol_idx = 0;
      lvl_regi.init_mov_vol_poses = { device_init_guess };
      lvl_regi.ref_frames = { 1 }; // This refers to multiview device
      {
        auto cmaes_regi = std::make_shared<CMAESRegi>();
        cmaes_regi->set_opt_vars(se3_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        using NormPDF = NormalDist1D<CoordScalar>;
        auto pen_fn = std::make_shared<Regi2D3DSE3EulerDecompPenaltyFn<NormPDF>>();
      
        cmaes_regi->set_pop_size(20);
        cmaes_regi->set_sigma({ 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5, 2.5, 5 });
        
        pen_fn->rot_x_pdf   = NormPDF(0, 2.5 * kDEG2RAD);
        pen_fn->rot_y_pdf   = NormPDF(0, 2.5 * kDEG2RAD);
        pen_fn->rot_z_pdf   = NormPDF(0, 2.5 * kDEG2RAD);
        pen_fn->trans_x_pdf = NormPDF(0, 2.5);
        pen_fn->trans_y_pdf = NormPDF(0, 2.5);
        pen_fn->trans_z_pdf = NormPDF(0, 5);

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

        lvl_regi.regi = cmaes_regi;
      }
    }
  }
  vout << std::endl << "Multi-view Device registration ..." << std::endl;
  regi.run();
  regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();
  if(kSAVE_REGI_DEBUG){
    size_type img_num = regi.fixed_cam_imgs.size();
    std::vector<SingleProjData<PixelScalar>> tmp_pd_list(img_num);
    for (size_type v = 0; v < img_num; ++v)
    {
      tmp_pd_list[v].img = regi.fixed_cam_imgs[v].img;
      tmp_pd_list[v].cam = regi.fixed_cam_imgs[v].cam;
    }
    WriteRegi2D3DMultiLevelDebug(regi.debug_info, output_path+"/device_mv_debug.h5",
                                std::make_tuple(VolPtr(), LabelVolPtr(), tmp_pd_list));
  }

  FrameTransform device_regi_xform = regi.cur_cam_to_vols[0];
  
  regi.levels.resize(2);
  regi.init_cam_to_vols = {device_regi_xform, init_femur_xform_devicefid };
  regi.debug_info.regi_names = {{"Femur-coarse"}, {"Femur-fine"}};
  regi.fixed_cam_imgs.resize(num_views);
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    regi.fixed_cam_imgs[view_idx].img = projs_st_to_regi[view_idx].img;
  }

  // Femur Registration-Coarse
  {
    auto& lvl = regi.levels[0];
    
    lvl.ds_factor = 0.125;
    lvl.fixed_imgs_to_use.resize(num_views);
    std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);
    auto rc = std::make_shared<RayCasterLineIntGPU>(gpu_prefs.ctx, gpu_prefs.queue);
      
    lvl.ray_caster = rc;
    {
      lvl.sim_metrics.resize(num_views);
      for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
      {
        auto sm = std::make_shared<PatchGradNCCImageSimilarityMetricGPU>(gpu_prefs.ctx, gpu_prefs.queue);
        sm->set_setup_vienna_cl_ctx(false);
        sm->set_smooth_img_before_sobel_kernel_radius(5);
        sm->set_patch_radius(std::lround(lvl.ds_factor * 41));
        sm->set_patch_stride(1);
        lvl.sim_metrics[view_idx] = sm;
      }
    }

    using FoldNormPDF = FoldedNormalPDF<CoordScalar>;
    using PenFn       = Regi2D3DSE3MagPenaltyFn<FoldNormPDF, FoldNormPDF>;

    lvl.regis.resize(1);
    {
      auto& lvl_regi = lvl.regis[0];
      lvl_regi.mov_vols    = { 1 }; // This refers to device (moving)
      lvl_regi.static_vols = { }; // This refers to femur and femur(static)
              
      auto femur_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      femur_init_guess->vol_idx = 1;
        
      // index into regi.ref_frames
      lvl_regi.ref_frames = { 2 }; // This refers to device
      lvl_regi.init_mov_vol_poses = { femur_init_guess };

      // Set CMAES parameters
      auto cmaes_regi = std::make_shared<CMAESRegi>();
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(0.01);
      cmaes_regi->set_opt_obj_fn_tol(0.01);
    
      cmaes_regi->set_pop_size(100);
      cmaes_regi->set_sigma({ 2 * kDEG2RAD, 2 * kDEG2RAD, 2 * kDEG2RAD, 1, 1, 1});
    
      auto pen_fn = std::make_shared<PenFn>();
      pen_fn->rot_pdfs_per_obj.assign(1, FoldNormPDF(3 * kDEG2RAD, 3 * kDEG2RAD));
      pen_fn->trans_pdfs_per_obj.assign(1, FoldNormPDF(2.5, 2.5));

      cmaes_regi->set_penalty_fn(pen_fn);
      cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

      lvl_regi.regi = cmaes_regi;
    }
  }

  // Femur Registration-fine
  {
    auto& lvl = regi.levels[1];
    lvl.ds_factor = 0.25;
    lvl.fixed_imgs_to_use.resize(num_views);
    std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);
    auto rc = std::make_shared<RayCasterLineIntGPU>(gpu_prefs.ctx, gpu_prefs.queue);
    lvl.ray_caster = rc;
    {
      lvl.sim_metrics.resize(num_views);
      for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
      {
        auto sm = std::make_shared<PatchGradNCCImageSimilarityMetricGPU>(gpu_prefs.ctx, gpu_prefs.queue);
        sm->set_setup_vienna_cl_ctx(false);
        sm->set_smooth_img_before_sobel_kernel_radius(5);
        sm->set_patch_radius(std::lround(lvl.ds_factor * 41));
        sm->set_patch_stride(1);
        lvl.sim_metrics[view_idx] = sm;
      }
    }

    using FoldNormPDF = FoldedNormalPDF<CoordScalar>;
    using PenFn       = Regi2D3DSE3MagPenaltyFn<FoldNormPDF, FoldNormPDF>;

    lvl.regis.resize(1);
    {
      auto& lvl_regi = lvl.regis[0];

      lvl_regi.mov_vols    = { 1 }; // This refers to femur (moving)
      lvl_regi.static_vols = { };
              
      auto femur_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      femur_init_guess->vol_idx = 1;
      
      // index into regi.ref_frames
      lvl_regi.ref_frames = { 2 }; // This refers to femur (moving)
      lvl_regi.init_mov_vol_poses = { femur_init_guess };

      auto bobyqa_regi = std::make_shared<BOBYQARegi>();
      {
        bobyqa_regi->set_opt_vars(se3_vars);
        bobyqa_regi->set_opt_x_tol(0.0001);
        bobyqa_regi->set_opt_obj_fn_tol(0.0001);
        bobyqa_regi->set_bounds({ 1.5 * kDEG2RAD, 1.5 * kDEG2RAD, 1.5 * kDEG2RAD,
                                  0.5, 0.5, 0.5 });
        
        lvl_regi.regi = bobyqa_regi;
      }
    }
  }

  regi.run();
  regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();
  if(kSAVE_REGI_DEBUG){
    size_type img_num = regi.fixed_cam_imgs.size();
    std::vector<SingleProjData<PixelScalar>> tmp_pd_list(img_num);
    for (size_type v = 0; v < img_num; ++v)
    {
      tmp_pd_list[v].img = regi.fixed_cam_imgs[v].img;
      tmp_pd_list[v].cam = regi.fixed_cam_imgs[v].cam;
    }
    WriteRegi2D3DMultiLevelDebug(regi.debug_info, output_path+"/femur_mv_debug.h5",
                                std::make_tuple(VolPtr(), LabelVolPtr(), tmp_pd_list));
  }

  vout << "saving transformations..." << std::endl;
  FrameTransform femur_regi_xform = regi.cur_cam_to_vols[1];
  
  WriteITKAffineTransform(output_path + "/femur_regi_xform.h5", femur_regi_xform);
  WriteITKAffineTransform(output_path + "/device_regi_xform.h5", device_regi_xform);
  
  {
    std::vector<SingleProjData<PixelScalar>> tmp_pd_list(regi.fixed_cam_imgs.size());
    
    for (size_type v = 0; v < regi.fixed_cam_imgs.size(); ++v)
    {
      tmp_pd_list[v].img = regi.fixed_cam_imgs[v].img;
      tmp_pd_list[v].cam = regi.fixed_cam_imgs[v].cam;
    }
    
    WriteProjData(output_path+"/proj_data.xml", tmp_pd_list);
  }
  
  {
    FrameTransform reproj_femur_xform = femur_regi_xform * init_device_cam_to_vols;
    RayCaster ray_caster;
    ray_caster.set_camera_model(default_cam);
    ray_caster.use_proj_store_replace_method();
    ray_caster.set_volume(femur_vol);
    ray_caster.set_num_projs(1);
    ray_caster.allocate_resources();
    ray_caster.xform_cam_to_itk_phys(0) = reproj_femur_xform;
    ray_caster.compute(0);
    
    WriteITKImageRemap8bpp(ray_caster.proj(0).GetPointer(), output_path + "/femur_repos.png");
  }
  
  vout << "performing device reprojection ..." << std::endl;
  
  {
    FrameTransform drill_rotcen_ref = FrameTransform::Identity();
    {
      auto drillref_fcsv = device_fcsv.find("RotCenter");
      Pt3 drill_rotcen_pt;
      
      if (drillref_fcsv != device_fcsv.end()){
        drill_rotcen_pt = drillref_fcsv->second;
      }
      else{
        std::cout << "ERROR: NOT FOUND DRILL REF PT" << std::endl;
      }
      
      drill_rotcen_ref.matrix()(0,3) = -drill_rotcen_pt[0];
      drill_rotcen_ref.matrix()(1,3) = -drill_rotcen_pt[1];
      drill_rotcen_ref.matrix()(2,3) = -drill_rotcen_pt[2];
    }
    
    FrameTransform regi_UReef_xform, repos_UReef_xform;
    {
      const std::string src_ureef_path          = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/Registration/UR_kinsFeb17/03/ur_eef.h5";
      H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
      H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
      H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
      std::vector<float> UReef_tracker          = ReadVectorH5<float>("TranformParameters", ureef_group0);
      regi_UReef_xform                         = ConvertSlicerToITK(UReef_tracker);
    }
    
    {
      const std::string src_ureef_path          = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/Registration/UR_kinsFeb17/04/ur_eef.h5";
      H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
      H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
      H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
      std::vector<float> UReef_tracker          = ReadVectorH5<float>("TranformParameters", ureef_group0);
       repos_UReef_xform                       = ConvertSlicerToITK(UReef_tracker);
    }
    
    const std::string handeye_X_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/Handeye/AXYB_output/devicehandeye_X.h5";
    FrameTransform handeye_regi_X;
    ReadITKAffineTransformFromFile(handeye_X_path, &handeye_regi_X);
    
    FrameTransform ref_device_xform = device_regi_xform;
      
    FrameTransform reproj_device_xform = drill_rotcen_ref.inverse() * handeye_regi_X.inverse() * repos_UReef_xform.inverse() * regi_UReef_xform * handeye_regi_X * drill_rotcen_ref * ref_device_xform * init_device_cam_to_vols;
    
    RayCaster ray_caster;
    ray_caster.set_camera_model(default_cam);
    ray_caster.use_proj_store_replace_method();
    ray_caster.set_volume(device_vol);
    ray_caster.set_num_projs(1);
    ray_caster.allocate_resources();
    ray_caster.xform_cam_to_itk_phys(0) = reproj_device_xform;
    ray_caster.compute(0);
    
    WriteITKImageRemap8bpp(ray_caster.proj(0).GetPointer(), output_path + "/drill_repos.png");
    
    auto repos_real_ptr = ReadITKImageFromDisk<Proj>("/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/Registration/RealXray_dcmFeb17/15");
    WriteITKImageRemap8bpp(repos_real_ptr.GetPointer(), output_path + "/real_repos.png");
  }
  
  vout << "exiting..." << std::endl;
  return kEXIT_VAL_SUCCESS;
}

