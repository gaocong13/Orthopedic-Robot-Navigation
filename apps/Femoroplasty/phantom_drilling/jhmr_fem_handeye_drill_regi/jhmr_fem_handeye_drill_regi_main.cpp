
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

#include "bigssMath.h"

using namespace jhmr;
using namespace jhmr::fem;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;
constexpr bool kSAVE_REGI_DEBUG = true;
using SimMetGPU = PatchGradNCCImageSimilarityMetricGPU;
using CMAESRegi           = Regi2D3DCMAES<MultiLevelMultiObjRegi::RayCaster,
MultiLevelMultiObjRegi::SimMetric>;

typedef RayCasterLineIntGPU::ImageVolumeType ImageVolumeType;

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
  #ifdef JHMR_HAS_OPENCL
    typedef CameraRayCasterGPULineIntegral RayCaster;
  #else
    typedef CameraRayCasterCPULineIntegral<float> RayCaster;
  #endif
  typedef RayCaster::CameraModelType CamModel;
  
  ProgOpts po;

  jhmrPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Compute 3D Polaris Position of Snake Tip Jig");
  po.set_arg_usage("<Source Bayview H5 root path> <Bayview H5 Debug file root path> <Bayview slicer root path> <Experiment list file path> <Result file path> <Top landmark path> <result path>");
  po.set_min_num_pos_args(4);
  
  po.add("verbose", 'v', ProgOpts::kSTORE_TRUE, "verbose", "Verbose logging to stdout")
    << false;  // default to non-verbose mode
  
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
  
  const std::string landmark2d_root_path    = po.pos_args()[0];  // 2D Landmark root path
  const std::string drill_3d_fcsv_path      = po.pos_args()[1];  // 3D drill landmarks path
  const std::string exp_list_path           = po.pos_args()[2];  // Experiment list file path
  const std::string dicom_path              = po.pos_args()[3];  // Dicom image path
  const std::string root_slicer_path        = po.pos_args()[4];  // Slicer path
  const std::string output_path             = po.pos_args()[5];  // Output path
 
  vout << "default GPU prefs..." << std::endl;
  GPUPrefsXML gpu_prefs;
  
  std::cout << "reading drill BB landmarks from FCSV file..." << std::endl;
  auto drill_3d_fcsv = ReadFCSVFileNamePtMap<Pt3>(drill_3d_fcsv_path);
  FromMapConvertRASToLPS(drill_3d_fcsv.begin(), drill_3d_fcsv.end());
  
  const std::string drillvol_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/meta_data/Device_cropmore_CT.nii.gz";
  const std::string drillseg_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/meta_data/Device_cropmore_seg.nii.gz";
  const std::string handeye_X_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/Handeye/AXYB_output/devicehandeye_X.h5";
  const std::string drillref_fcsv_path = drill_3d_fcsv_path;
  const std::string ref_ID = "01";
  const std::string ref_device_xform_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/Handeye/output_Feb16/drill_regi_xform" + ref_ID + ".h5";
  bool est_from_landmark = true;
  
  FrameTransform ref_device_xform;
  ReadITKAffineTransformFromFile(ref_device_xform_path, &ref_device_xform);
  
  FrameTransform handeye_regi_X;
  ReadITKAffineTransformFromFile(handeye_X_path, &handeye_regi_X);
  
  FrameTransform refUReef_xform;
  {
    const std::string src_ureef_path          = root_slicer_path + "/" + ref_ID + "/ur_eef.h5";
    H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
    H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
    H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
    std::vector<float> UReef_tracker          = ReadVectorH5<float>("TranformParameters", ureef_group0);
    refUReef_xform                            = ConvertSlicerToITK(UReef_tracker);
  }
  
  std::cout << "reading drill rotation center ref landmark from FCSV file..." << std::endl;
  auto drillref_3dfcsv = ReadFCSVFileNamePtMap<Pt3>(drillref_fcsv_path);
  FromMapConvertRASToLPS(drillref_3dfcsv.begin(), drillref_3dfcsv.end());
  
  FrameTransform drill_rotcen_ref = FrameTransform::Identity();
  {
    auto drillref_fcsv = drillref_3dfcsv.find("RotCenter");
    Pt3 drill_rotcen_pt;
    
    if (drillref_fcsv != drillref_3dfcsv.end()){
      drill_rotcen_pt = drillref_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND DRILL REF PT" << std::endl;
    }
    
    drill_rotcen_ref.matrix()(0,3) = -drill_rotcen_pt[0];
    drill_rotcen_ref.matrix()(1,3) = -drill_rotcen_pt[1];
    drill_rotcen_ref.matrix()(2,3) = -drill_rotcen_pt[2];
  }
  
  LabelVolPtr drill_seg = ReadITKImageFromDisk<LabelVol>(drillseg_path);
  
  VolPtr drillvol_att;
  {
    vout << "reading drill volume..." << std::endl; // We only use the needle metal part
    auto drillvol_hu = ReadITKImageFromDisk<Vol>(drillvol_path);

    vout << "  HU --> Att. ..." << std::endl;
    auto hu2att = HounsfieldToLinearAttenuationFilter<Vol>::New();
    hu2att->SetInput(drillvol_hu);
    hu2att->Update();
    
    drillvol_att = hu2att->GetOutput();
  }
  
  const LabelScalar drill_label = 1;
  VolPtr drill_vol = ApplyMaskToITKImage(drillvol_att.GetPointer(), drill_seg.GetPointer(), drill_label, PixelScalar(0), true);
  
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
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> drill_singleview_regi_ref_frame;
    
  {
    vout << "setting up drill ref. frame..." << std::endl;
    // setup camera aligned reference frame, use metal drill volume center point as the origin
  
    itk::ContinuousIndex<double,3> center_idx;

    auto drill_fcsv_rotc = drill_3d_fcsv.find("RotCenter");
    Pt3 drill_rotcenter;

    if (drill_fcsv_rotc != drill_3d_fcsv.end()){
      drill_rotcenter = drill_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND DRILL ROT CENTER" << std::endl;
    }

    drill_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    drill_singleview_regi_ref_frame->vol_idx = 0;
    drill_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = drill_rotcenter[0];
    drill_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = drill_rotcenter[1];
    drill_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = drill_rotcenter[2];
  }
  
  if(lineNumber!=exp_ID_list.size()) throw std::runtime_error("Exp ID list size mismatch!!!");
  
  const CamModel default_cam = NaiveCamModelFromCIOSFusion(
  MakeNaiveCIOSFusionMetaDR(), true).cast<CoordScalar>();
  
  bool is_first_view = true;
  
  for(int idx = 0; idx < lineNumber; ++idx)
  {
    const std::string exp_ID                = exp_ID_list[idx];
    const std::string drillbb_2d_fcsv_path  = landmark2d_root_path + "/" + exp_ID + ".fcsv";
    const std::string img_path              = dicom_path + "/" + exp_ID;
    
    std::cout << "Running..." << exp_ID << std::endl;
    
    ProjPreProc<PixelScalar> proj_pre_proc;
    proj_pre_proc.input_projs.resize(1);
    
    std::vector<CIOSFusionDICOMInfo> drillcios_metas(1);
    {
      std::tie(proj_pre_proc.input_projs[0].img, drillcios_metas[0]) =
                                                      ReadCIOSFusionDICOM<PixelScalar>(img_path);
      proj_pre_proc.input_projs[0].cam = NaiveCamModelFromCIOSFusion(
                                                      drillcios_metas[0], true).cast<CoordScalar>();
    }
    
    if(est_from_landmark)
    {
      auto drillbb_2d_fcsv = ReadFCSVFileNamePtMap<Pt3>(drillbb_2d_fcsv_path);
      FromMapConvertRASToLPS(drillbb_2d_fcsv.begin(), drillbb_2d_fcsv.end());
      
      jhmrASSERT(drillbb_2d_fcsv.size() > 3);

      LandMap2 drillproj_lands;
      /*
      for (const auto& fcsv_kv : bb_fcsv)
      {
        drillproj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
      }
       */
      
      {
        UpdateLandmarkMapForCIOSFusion(drillcios_metas[0], drillbb_2d_fcsv.begin(), drillbb_2d_fcsv.end());
        auto& drillproj_lands = proj_pre_proc.input_projs[0].landmarks;
        drillproj_lands.reserve(drillbb_2d_fcsv.size());

        for (const auto& fcsv_kv : drillbb_2d_fcsv)
        {
          drillproj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
        }
      }
    }
    
    proj_pre_proc();
    auto& projs_to_regi = proj_pre_proc.output_projs;
    
    {
      std::vector<SingleProjData<PixelScalar>> tmp_pd_list(1);
      {
        tmp_pd_list[0].img = projs_to_regi[0].img;
        tmp_pd_list[0].cam = projs_to_regi[0].cam;
      }
      WriteProjData(output_path+"/proj_dr_data" + exp_ID + ".xml", tmp_pd_list);
    }
    
    FrameTransform init_cam_to_drill;
    if(est_from_landmark){
      init_cam_to_drill = EstCamToWorldBruteForcePOSITCMAESRefine(projs_to_regi[0].cam, projs_to_regi[0].landmarks, drill_3d_fcsv);
    }
    else
    {
      const std::string src_ureef_path          = root_slicer_path + "/" + exp_ID + "/ur_eef.h5";
      H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
      H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
      H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
      std::vector<float> UReef_tracker          = ReadVectorH5<float>("TranformParameters", ureef_group0);
      FrameTransform UReef_xform                = ConvertSlicerToITK(UReef_tracker);
      
      init_cam_to_drill = drill_rotcen_ref.inverse() * handeye_regi_X.inverse() * UReef_xform.inverse() * refUReef_xform * handeye_regi_X * drill_rotcen_ref * ref_device_xform;
    }
    
    WriteITKAffineTransform(output_path + "/drill_init_xform" + exp_ID + ".h5", init_cam_to_drill);
    
    MultiLevelMultiObjRegi regi;

    regi.save_debug_info           = kSAVE_REGI_DEBUG;
    regi.debug_info.vol_path       = drillvol_path;
    regi.debug_info.label_vol_path  = drillseg_path;
    regi.debug_info.labels_used = { 1 };
    regi.vols = { drill_vol };
    regi.vol_names = { "Drill"};
      
    regi.ref_frames = { drill_singleview_regi_ref_frame };

    regi.fixed_cam_imgs.resize(1);
    
    const size_type view_idx = 0;
    regi.fixed_cam_imgs[view_idx].img = projs_to_regi[view_idx].img;
    regi.fixed_cam_imgs[view_idx].cam = projs_to_regi[view_idx].cam;
    
    regi.levels.resize(1);
    
    regi.debug_info.regi_names = { {"Drill"} };
    regi.init_cam_to_vols = { init_cam_to_drill };
    
    drill_singleview_regi_ref_frame->cam_extrins = regi.fixed_cam_imgs[view_idx].cam.extrins;
    
    auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();
    
    auto& lvl = regi.levels[0];
    
    lvl.ds_factor = 0.25;
    
    lvl.fixed_imgs_to_use = { view_idx };

    auto rc = std::make_shared<RayCasterLineIntGPU>(gpu_prefs.ctx, gpu_prefs.queue);
    lvl.ray_caster = rc;
    {
      auto sm = std::make_shared<SimMetGPU>(gpu_prefs.ctx, gpu_prefs.queue);
      sm->set_setup_vienna_cl_ctx(is_first_view);
      is_first_view = false;
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
        cmaes_regi->set_sigma({ 20 * kDEG2RAD, 20 * kDEG2RAD, 20 * kDEG2RAD, 50, 50, 50 });
        
        pen_fn->rot_x_pdf   = NormPDF(0, 15 * kDEG2RAD);
        pen_fn->rot_y_pdf   = NormPDF(0, 15 * kDEG2RAD);
        pen_fn->rot_z_pdf   = NormPDF(0, 15 * kDEG2RAD);
        pen_fn->trans_x_pdf = NormPDF(0, 25);
        pen_fn->trans_y_pdf = NormPDF(0, 25);
        pen_fn->trans_z_pdf = NormPDF(0, 25);

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);
        
        lvl_regi_coarse.regi = cmaes_regi;
      }
      vout << std::endl << "First view spine coarse registration ..." << std::endl;
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
        WriteRegi2D3DMultiLevelDebug(regi.debug_info, output_path+"/drill_regi_debug" + exp_ID + ".h5",
                                    std::make_tuple(VolPtr(), LabelVolPtr(), tmp_pd_list));
      }
    }
    
    WriteITKAffineTransform(output_path + "/drill_regi_xform" + exp_ID + ".h5", regi.cur_cam_to_vols[0]);
    
    {
      RayCaster ray_caster;
      ray_caster.set_camera_model(default_cam);
      ray_caster.use_proj_store_replace_method();
      ray_caster.set_volume(drill_vol);
      ray_caster.set_num_projs(1);
      ray_caster.allocate_resources();
      ray_caster.xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[0];
      ray_caster.compute(0);
      
      WriteITKImageRemap8bpp(ray_caster.proj(0).GetPointer(), output_path + "/drill_reproj" + exp_ID + ".png");
      WriteITKImageRemap8bpp(proj_pre_proc.input_projs[0].img.GetPointer(), output_path + "/real" + exp_ID + ".png");
    }
  }
  return 0;
}

