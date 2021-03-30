
// STD
#include <iostream>
#include <vector>

// Boost
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/null.hpp>

#include <fmt/format.h>

// RayCasting
#include "jhmrItkConvertUtils.h"
#include "jhmrCameraRayCastingCPU.h"

#ifdef JHMR_HAS_OPENCL
#include "jhmrCameraRayCastingGPU.h"
#endif

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

using Cam                  = MultiLevelMultiObjRegi::CamImgPair::CamModel;
using Proj                 = MultiLevelMultiObjRegi::CamImgPair::ImageType;
using RayCasterLineIntGPU  = CameraRayCasterGPULineIntegral;
using CMAESRegi            = Regi2D3DCMAES<MultiLevelMultiObjRegi::RayCaster,
MultiLevelMultiObjRegi::SimMetric>;
using BOBYQARegi           = Regi2D3DBOBYQA<MultiLevelMultiObjRegi::RayCaster,
MultiLevelMultiObjRegi::SimMetric>;

using SimMetGPU            = PatchGradNCCImageSimilarityMetricGPU;
using SimMetCPU            = PatchGradNCCImageSimilarityMetricCPU<PixelScalar>;

using ExhaustiveRegi       = Regi2D3DExhaustive<MultiLevelMultiObjRegi::RayCaster,
MultiLevelMultiObjRegi::SimMetric>;

using FoldNormPDF          = FoldedNormalPDF<CoordScalar>;
using RotMagAndTransMagPen = Regi2D3DSE3MagPenaltyFn<FoldNormPDF>;

typedef RayCasterLineIntGPU::ImageVolumeType ImageVolumeType;

auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();
auto so3_vars = std::make_shared<SO3OptVarsLieAlg<double>>();

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

int main(int argc, char* argv[])
{
  // Set up the program options

  ProgOpts po;

  jhmrPROG_OPTS_SET_COMPILE_DATE(po);
  // allow negative numbers (e.g. leading with -) positional arguments
  //po.set_allow_unrecognized_flags(true);

  po.set_help("Example driver for a multi-view femur registration using femur as fiducial object.");
  po.set_arg_usage("<CT intensity 3D volume> <Femur segmentation volume> <Femur 2D landmark> <Femur 3D landmark> "
                   "<Output path> <Image Path>");
  po.set_min_num_pos_args(6);

  po.add("verbose", 'v', ProgOpts::kSTORE_TRUE, "verbose", "Verbose logging to stdout")
    << false;  // default to non-verbose mode
  po.add("debug", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "debug", "Save debug info")
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

  const std::string vol_path                   = po.pos_args()[0];
  const std::string seg_path                   = po.pos_args()[1];
  const std::string femurld_2d_fcsv_path       = po.pos_args()[2];  // 2D femur landmarks path
  const std::string femurld_3d_fcsv_path       = po.pos_args()[3];  // 3D femur landmarks path
  const std::string output_path                = po.pos_args()[4];
  const std::string img_path                   = po.pos_args()[5];
  
  std::vector<std::string> img_ID_list(po.pos_args().begin()+6, po.pos_args().end()); // First 3 are soft tissue; Last 3 are with drill
  
  const size_type num_views = 1;//Hard code
  const LabelScalar femur_label = 1;
  
  const CamModel default_cam = NaiveCamModelFromCIOSFusion(
  MakeNaiveCIOSFusionMetaDR(), true).cast<CoordScalar>();

  vout << "default GPU prefs..." << std::endl;
  GPUPrefsXML gpu_prefs;
  
  vout << "reading input CT volume..." << std::endl;
  auto vol_hu = ReadITKImageFromDisk<Vol>(vol_path);
  
  /*
  VolPtr femur_vol;
  {
    itk::ImageRegionIterator<ImageVolumeType> it_bone(vol_hu, vol_hu->GetRequestedRegion());
    it_bone.GoToBegin();
    while( !it_bone.IsAtEnd()){
      if(it_bone.Get() == 1){
        it_bone.Set(500);
      }
      else if(it_bone.Get() == 0){
        it_bone.Set(-1000);
      }
      ++it_bone;
    }

    vout << "  HU --> Att. ..." << std::endl;
    auto hu2att = HounsfieldToLinearAttenuationFilter<Vol>::New();
    hu2att->SetInput(vol_hu);
    hu2att->Update();
    
    femur_vol = hu2att->GetOutput();
  }
   */
  
  VolPtr vol_att;
  {
    vout << "  HU --> Att. ..." << std::endl;
    auto hu2att = HounsfieldToLinearAttenuationFilter<Vol>::New();
    hu2att->SetInput(vol_hu);
    hu2att->Update();
    
    vol_att = hu2att->GetOutput();
  }
  
  vout << "reading volume segmentation..." << std::endl;
  LabelVolPtr vol_seg = ReadITKImageFromDisk<LabelVol>(seg_path);

  vout << "extracting femur att. volume..." << std::endl;
  VolPtr femur_vol = ApplyMaskToITKImage(vol_att.GetPointer(), vol_seg.GetPointer(), femur_label, PixelScalar(0), true);
  
  vout << "reading femur 2d landmarks from FCSV file..." << std::endl;
  auto femurld_2d_fcsv = ReadFCSVFileNamePtMap<Pt3>(femurld_2d_fcsv_path);
  FromMapConvertRASToLPS(femurld_2d_fcsv.begin(), femurld_2d_fcsv.end());
  
  vout << "reading femur 3d Landmarks from FCSV file..." << std::endl;
  auto femurld_3d_fcsv = ReadFCSVFileNamePtMap<Pt3>(femurld_3d_fcsv_path);
  FromMapConvertRASToLPS(femurld_3d_fcsv.begin(), femurld_3d_fcsv.end());
  
  ProjPreProc<PixelScalar> proj_st_pre_proc;
  proj_st_pre_proc.input_projs.resize(num_views);
  
  vout << "reading 2D softtissue images..." << std::endl;
  std::vector<CIOSFusionDICOMInfo> cios_metas_st(num_views);
  
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    std::string img_path_ID = img_path + "/" + img_ID_list[view_idx];
    std::tie(proj_st_pre_proc.input_projs[view_idx].img, cios_metas_st[view_idx]) = ReadCIOSFusionDICOM<PixelScalar>(img_path_ID);
    /*
    Proj::SpacingType sim_res;
    sim_res[0] = 0.194;
    sim_res[1] = 0.194;
    std::string img_path_ID = img_path + "/" + img_ID_list[view_idx] + ".tif";
    proj_st_pre_proc.input_projs[view_idx].img = ReadITKImageFromDisk<Proj>(img_path_ID);
    proj_st_pre_proc.input_projs[view_idx].img->SetSpacing(sim_res);
    cios_metas_st[view_idx] = MakeNaiveCIOSFusionMetaDR();
    */
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

  vout << "running 2D preprocessing..." << std::endl;
  proj_st_pre_proc();

  auto& projs_st_to_regi = proj_st_pre_proc.output_projs;

  vout << "Extract initialization from simulation..." << std::endl;
  
  FrameTransform gt_xform = EstCamToWorldBruteForcePOSITCMAESRefine(proj_st_pre_proc.output_projs[0].cam, proj_st_pre_proc.output_projs[0].landmarks, femurld_3d_fcsv);
  //ReadITKAffineTransformFromFile(output_path + "/gt_xform.h5", &gt_xform);
  //ReadITKAffineTransformFromFile("/Users/gaocong/Documents/Research/Spine/Drill_Handeye/drill_pnp/drillpnp_xformOct01-003.h5", &drill_xform);
  
  FrameTransform init_cam_0_to_vol = gt_xform;

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> femur_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> femur_multiview_regi_ref_frame;
  {
    vout << "setting up femur ref. frame..." << std::endl;
    Pt3 femur_pt;

    auto check_label = [&vol_hu,&femur_pt,&femurld_3d_fcsv,femur_label] (const std::string& k)
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
        
        vol_hu->TransformPhysicalPointToIndex(tmp_itk_pt, tmp_itk_idx);
        
//        found = vol_seg->GetPixel(tmp_itk_idx) == femur_label;
        found = true;
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
  
  Timer tmr;
  tmr.start();
  
  MultiLevelMultiObjRegi regi;

  regi.save_debug_info           = kSAVE_REGI_DEBUG;
  regi.debug_info.vol_path       = vol_path;
  regi.debug_info.label_vol_path = seg_path;
  regi.debug_info.labels_used = { femur_label };
  regi.vols = { femur_vol };
  regi.vol_names = { "Femur"};
    
  regi.ref_frames = { femur_singleview_regi_ref_frame };

  regi.fixed_cam_imgs.resize(num_views);
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    regi.fixed_cam_imgs[view_idx].img = projs_st_to_regi[view_idx].img;
    regi.fixed_cam_imgs[view_idx].cam = projs_st_to_regi[view_idx].cam;
  }
    
  regi.levels.resize(1);

  FrameTransformList regi_cams_to_femur_vol(num_views);

  const size_type view_idx = 0;
  
  const bool is_first_view = view_idx == 0;

  {
    regi.debug_info.regi_names = { { "Femur" } };
    regi.init_cam_to_vols = { init_cam_0_to_vol };
  }

  femur_singleview_regi_ref_frame->cam_extrins = regi.fixed_cam_imgs[view_idx].cam.extrins;

  auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();

  auto& lvl = regi.levels[0];

  lvl.ds_factor = 0.25;

  lvl.fixed_imgs_to_use = { view_idx };

  auto rc = std::make_shared<RayCasterLineIntGPU>(gpu_prefs.ctx, gpu_prefs.queue);
  lvl.ray_caster = rc;
  {
    auto sm = std::make_shared<SimMetGPU>(gpu_prefs.ctx, gpu_prefs.queue);
    sm->set_setup_vienna_cl_ctx(is_first_view);
    sm->set_smooth_img_before_sobel_kernel_radius(1);
    sm->set_patch_radius(std::lround(lvl.ds_factor * 41));
    sm->set_patch_stride(5);
    lvl.sim_metrics = { sm };
  }

  lvl.regis.resize(1);

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
  
    cmaes_regi->set_pop_size(150);
    cmaes_regi->set_sigma({ 30 * kDEG2RAD, 30 * kDEG2RAD, 30 * kDEG2RAD, 50, 50, 100 });
    
    pen_fn->rot_x_pdf   = NormPDF(0, 30 * kDEG2RAD);
    pen_fn->rot_y_pdf   = NormPDF(0, 30 * kDEG2RAD);
    pen_fn->rot_z_pdf   = NormPDF(0, 30 * kDEG2RAD);
    pen_fn->trans_x_pdf = NormPDF(0, 40);
    pen_fn->trans_y_pdf = NormPDF(0, 40);
    pen_fn->trans_z_pdf = NormPDF(0, 75);

    cmaes_regi->set_penalty_fn(pen_fn);
    cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);
    
    lvl_regi_coarse.regi = cmaes_regi;
  }
  vout << std::endl << "Single view femur registration ..." << std::endl;
  regi.run();
  regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

  if(kSAVE_REGI_DEBUG)
  {
    size_type img_num = regi.fixed_cam_imgs.size();
    std::vector<SingleProjData<PixelScalar>> tmp_pd_list(img_num);
    for (size_type v = 0; v < img_num; ++v)
    {
      tmp_pd_list[v].img = regi.fixed_cam_imgs[v].img;
      tmp_pd_list[v].cam = regi.fixed_cam_imgs[v].cam;
    }
    WriteRegi2D3DMultiLevelDebug(regi.debug_info, output_path+"/femur_debug.h5",
                                std::make_tuple(VolPtr(), LabelVolPtr(), tmp_pd_list));
  }
  
  WriteITKAffineTransform(output_path + "/femur_regi_xform" + img_ID_list[view_idx] + ".h5", regi.cur_cam_to_vols[0]);
  
  std::vector<SingleProjData<PixelScalar>> tmp_pd_list(regi.fixed_cam_imgs.size());
  
  for (size_type v = 0; v < regi.fixed_cam_imgs.size(); ++v)
  {
    tmp_pd_list[v].img = regi.fixed_cam_imgs[v].img;
    tmp_pd_list[v].cam = regi.fixed_cam_imgs[v].cam;
  }
  
  WriteProjData(output_path+"/proj_fem_data.xml", tmp_pd_list);
  
  {
    RayCasterLineIntGPU ray_caster;
    ray_caster.set_camera_model(default_cam);
    ray_caster.use_proj_store_replace_method();
    ray_caster.set_volume(femur_vol);
    ray_caster.set_num_projs(1);
    ray_caster.allocate_resources();
    ray_caster.xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[0];
    ray_caster.compute(0);
    
    WriteITKImageRemap8bpp(ray_caster.proj(0).GetPointer(), output_path + "/femur_reproj.png");
    
    WriteITKImageRemap8bpp(proj_st_pre_proc.input_projs[0].img.GetPointer(), output_path + "/real.png");
  }
  vout << "exiting..." << std::endl;
  return kEXIT_VAL_SUCCESS;
}

