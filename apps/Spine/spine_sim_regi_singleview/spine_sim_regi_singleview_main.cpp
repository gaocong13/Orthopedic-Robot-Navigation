
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
#include "xregHUToLinAtt.h"
#include "xregImageAddPoissonNoise.h"
#include "xregSampleUtils.h"
#include "xregSampleUniformUnitVecs.h"
#include "xregRotUtils.h"
#include "xregRigidUtils.h"

#include "xregPAODrawBones.h"
#include "xregRigidUtils.h"

#include "bigssMath.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVersor.h"
#include "itkChangeInformationImageFilter.h"

#include "xregRecomposeVertebraes.h"

using namespace xreg;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;
constexpr bool kSAVE_REGI_DEBUG = false;

using size_type = std::size_t;
using CamModelList = std::vector<CameraModel>;

using Pt3         = Eigen::Matrix<CoordScalar,3,1>;
using Pt2         = Eigen::Matrix<CoordScalar,2,1>;

FrameTransform Delta_rand_matrix(float rot_mag, float trans_mag)
{
  std::mt19937 rng_eng;
  SeedRNGEngWithRandDev(&rng_eng);
  // for sampling random rotation axes and random translation directions
  UniformOnUnitSphereDist unit_vec_dist(3);

  // for sampling random rotation angles (in APP w/ origin at FH)
  std::uniform_real_distribution<CoordScalar> rot_ang_dist(-rot_mag*kDEG2RAD, rot_mag*kDEG2RAD);

  // for sampling random translation magnitudes (in APP)
  std::uniform_real_distribution<CoordScalar> trans_mag_dist(-trans_mag, trans_mag);

  FrameTransform delta_ref = FrameTransform::Identity();
  {
    // add some noise in the APP coordinate frame
    Pt3 so3 = unit_vec_dist(rng_eng);
    so3 *= rot_ang_dist(rng_eng);

    Eigen::Matrix<CoordScalar,3,3> rot_mat;
    rot_mat = ExpSO3(so3);
    delta_ref.matrix().block(0,0,3,3) = rot_mat;

    Pt3 trans = unit_vec_dist(rng_eng);
    trans *= trans_mag_dist(rng_eng);
    delta_ref.matrix().block(0,3,3,1) = trans;
  }

  return delta_ref;
}

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Simulation of Single-view Spine Registration");
  po.set_arg_usage("< meta data path > < output path >");
  po.set_min_num_pos_args(2);

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

  const std::string meta_data_path          = po.pos_args()[0];  // 3D spine landmarks path
  const std::string output_path             = po.pos_args()[1];  // Output path

  const std::string spinevol_path = meta_data_path + "/Spine21-2512_CT_crop.nrrd";
  const std::string spineseg_path = meta_data_path + "/Sheetness_seg_crop_mapped.nrrd";
  const std::string spine_gt_xform_path = meta_data_path + "/spine_gt_xform.h5";
  const std::string device_gt_xform_path = meta_data_path + "/device_gt_xform.h5";
  const std::string spine_3d_fcsv_path = meta_data_path + "/Spine_3D_landmarks.fcsv";

  const std::string device_3d_fcsv_path    = meta_data_path + "/Device3Dlandmark.fcsv";
  const std::string device_3d_bb_fcsv_path = meta_data_path + "/Device3Dbb.fcsv";
  const std::string devicevol_path         = meta_data_path + "/Device_crop_CT.nii.gz";
  const std::string deviceseg_path         = meta_data_path + "/Device_crop_seg.nii.gz";

  const float rot_mag_leftright = 10.; // Vertebrae axis Y in degrees
  const float rot_mag_backforth = 10.; // Vertebrae axis X in degrees
  const float rot_mag_tilt = 5.;       // Vertebrae axis Z in degrees
  const float rot_mag_device = 10;     // Device initialization random rotation in degrees;
  const float trans_mag_device = 10;   // Device initialization random translation in mm;
  const float rot_mag_spine = 10;      // Spine initialization random rotation in degrees;
  const float trans_mag_spine = 10;    // Spine initialization random translation in mm;

  vout << "reading spine anatomical landmarks from FCSV file..." << std::endl;
  auto spine_3d_fcsv = ReadFCSVFileNamePtMap(spine_3d_fcsv_path);
  ConvertRASToLPS(&spine_3d_fcsv);

  vout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_fcsv = ReadFCSVFileNamePtMap(device_3d_fcsv_path);
  ConvertRASToLPS(&device_3d_fcsv);

  const bool use_seg = true;
  auto spine_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(spineseg_path);

  vout << "reading spine volume..." << std::endl; // We only use the needle metal part
  auto spinevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(spinevol_path);

  // Resample Spine Vertebraes Popses
  std::mt19937 rng_eng;
  SeedRNGEngWithRandDev(&rng_eng);
  std::uniform_real_distribution<float> rot_angX_dist(-rot_mag_backforth, rot_mag_backforth);
  std::uniform_real_distribution<float> rot_angY_dist(-rot_mag_leftright, rot_mag_leftright);
  std::uniform_real_distribution<float> rot_angZ_dist(-rot_mag_tilt, rot_mag_tilt);

/*
  using FilterType = itk::ChangeInformationImageFilter<RayCaster::Vol>;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(spinevol_hu.GetPointer());

  filter->SetOutputOrigin(spine_seg->GetOrigin());
  filter->SetOutputSpacing(spine_seg->GetSpacing());
  filter->ChangeOriginOn();
  filter->ChangeSpacingOn();

  filter->UpdateOutputInformation();

  std::cout << spine_seg->GetOrigin() << std::endl;
  std::cout << filter->GetOutput()->GetOrigin() << std::endl;

  WriteITKImageToDisk(filter->GetOutput(), meta_data_path + "/Spine21-2512_CT_crop_origin.nii.gz");

  return 0;
*/
  {
    spinevol_hu->SetOrigin(spine_seg->GetOrigin());
    spinevol_hu->SetSpacing(spine_seg->GetSpacing());
  }

  vout << "  HU --> Att. ..." << std::endl;
  auto spinevol_att = HUToLinAtt(spinevol_hu.GetPointer());

  unsigned char spine_label = 1;
  auto spine_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), spine_label, float(0), true);

  auto device_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(deviceseg_path);

  vout << "reading device volume..." << std::endl; // We only use the needle metal part
  auto devicevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(devicevol_path);

  vout << "  HU --> Att. ..." << std::endl;

  auto devicevol_att = HUToLinAtt(devicevol_hu.GetPointer());

  unsigned char device_label = 1;

  auto device_vol = ApplyMaskToITKImage(devicevol_att.GetPointer(), device_seg.GetPointer(), device_label, float(0), true);

  const FrameTransform gt_cam_wrt_spine = ReadITKAffineTransformFromFile(spine_gt_xform_path);
  const FrameTransform gt_cam_wrt_device = ReadITKAffineTransformFromFile(device_gt_xform_path);

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> spine_singleview_regi_ref_frame;
  FrameTransform spine_ref_frame = FrameTransform::Identity();
  {
    vout << "setting up spine ref. frame..." << std::endl;
    // setup camera aligned reference frame, use metal spine volume center point as the origin

    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert3-cen");
    Pt3 spine_rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      spine_rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND spine spine head center" << std::endl;
    }

    spine_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    spine_singleview_regi_ref_frame->vol_idx = 0;
    spine_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = spine_rotcenter[0];
    spine_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = spine_rotcenter[1];
    spine_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = spine_rotcenter[2];
    spine_ref_frame(0, 3) = -spine_rotcenter[0];
    spine_ref_frame(1, 3) = -spine_rotcenter[1];
    spine_ref_frame(2, 3) = -spine_rotcenter[2];
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> device_singleview_regi_ref_frame;
  FrameTransform device_ref_frame = FrameTransform::Identity();
  {
    vout << "setting up device ref. frame..." << std::endl;
    // setup camera aligned reference frame, use metal device volume center point as the origin

    itk::ContinuousIndex<double,3> center_idx;

    auto device_fcsv_rotc = device_3d_fcsv.find("RotCenter");
    Pt3 device_rotcenter;

    if (device_fcsv_rotc != device_3d_fcsv.end()){
      device_rotcenter = device_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND DRILL ROT CENTER" << std::endl;
    }

    device_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    device_singleview_regi_ref_frame->vol_idx = 1;
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = device_rotcenter[0];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = device_rotcenter[1];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = device_rotcenter[2];
    device_ref_frame(0, 3) = -device_rotcenter[0];
    device_ref_frame(1, 3) = -device_rotcenter[1];
    device_ref_frame(2, 3) = -device_rotcenter[2];
  }

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  CamModelList cams;
  cams.reserve(1);
  cams.push_back(default_cam);

  ProjDataF32 proj_spine;
  ProjDataF32 proj_device;

  for (size_type device_flag = 0; device_flag < 2; device_flag++)
  {
    auto ray_caster = LineIntRayCasterFromProgOpts(po);
    ray_caster->set_camera_models(cams);
    ray_caster->set_volumes({devicevol_att, spinevol_att});
    // ray_caster->set_ray_step_size(0.5);
    ray_caster->set_num_projs(1);
    ray_caster->allocate_resources();

    if (device_flag == 1)
    {
      ray_caster->use_proj_store_replace_method();
      ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_device);
      ray_caster->compute(0);
      ray_caster->use_proj_store_accum_method();
    }

    ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_spine);
    ray_caster->compute(1);

    vout << "projecting..." << device_flag << std::endl;

    if (device_flag == 1)
    {
      proj_device.img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 5000);
      proj_device.cam = default_cam;
    }
    else
    {
      proj_spine.img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 5000);
      proj_spine.cam = default_cam;
    }
  }

  if (kSAVE_REGI_DEBUG )
  {
    vout << "preprocessing bone data..." << std::endl;
    WriteITKImageRemap8bpp(proj_spine.img.GetPointer(), output_path + "/spine.png");

    vout << "preprocessing snake data..." << std::endl;
    WriteITKImageRemap8bpp(proj_device.img.GetPointer(), output_path + "/device.png");
  }

  for(int idx=0; idx < 1000; ++idx)
  {
    const std::string exp_ID = fmt::format("{:04d}", idx);
    std::cout << "Running..." << exp_ID << std::endl;

    const std::string dst_h5_path = output_path + "/" + exp_ID + ".h5";
    H5::H5File dst_h5(dst_h5_path, H5F_ACC_TRUNC);

    const float rot_angX_rand = rot_angX_dist(rng_eng);
    const float rot_angY_rand = rot_angY_dist(rng_eng);
    const float rot_angZ_rand = rot_angZ_dist(rng_eng);

    vout << "Sampling left/right:" << rot_angY_rand << " back/forth:" << rot_angX_rand << " tilt:" << rot_angZ_rand << std::endl;
    RecomposeVertebrae recomp_vertb(meta_data_path, rot_angX_rand, rot_angY_rand, rot_angZ_rand);
    auto resample_vert = recomp_vertb.ResampleVertebrae(vout);

    const std::string resample_vert_vol_path = output_path + "/reample_vert" + exp_ID + ".nii.gz";
    WriteITKImageToDisk(resample_vert.GetPointer(), resample_vert_vol_path);

    FrameTransform init_cam_wrt_spine = spine_ref_frame.inverse() * Delta_rand_matrix(rot_mag_spine, trans_mag_spine) *  spine_ref_frame * gt_cam_wrt_spine;
    FrameTransform init_cam_wrt_device = device_ref_frame.inverse() * Delta_rand_matrix(rot_mag_device, trans_mag_device) * device_ref_frame * gt_cam_wrt_device;

    ProjDataF32List proj_spine_list = { proj_spine };
    ProjDataF32List proj_device_list = { proj_device };

    H5::Group dst_spine_g = dst_h5.createGroup("proj-spine");
    WriteProjDataH5(proj_spine, &dst_spine_g);
    H5::Group dst_device_g = dst_h5.createGroup("proj-device");
    WriteProjDataH5(proj_device, &dst_device_g);

    MultiLevelMultiObjRegi regi;

    vout << "Setting up registration..." << std::endl;
    regi.set_debug_output_stream(vout, verbose);
    regi.set_save_debug_info(kSAVE_REGI_DEBUG);
    regi.vols = { resample_vert, device_vol };
    regi.vol_names = { "spine", "device" };

    regi.ref_frames = { spine_singleview_regi_ref_frame, device_singleview_regi_ref_frame };

    const size_type view_idx = 0;

    regi.fixed_proj_data = proj_spine_list;

    regi.levels.resize(1);

    regi.init_cam_to_vols = { init_cam_wrt_spine, init_cam_wrt_device };

    spine_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;
    device_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;

    auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();

    // Spine Single-view Registration
    {
      auto& lvl = regi.levels[0];

      lvl.ds_factor = 0.25;

      lvl.fixed_imgs_to_use = { view_idx };

      lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

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
        auto& lvl_regi_coarse = lvl.regis[0];
        lvl_regi_coarse.mov_vols = { 0 };
        lvl_regi_coarse.static_vols = { };

        auto init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        init_guess->vol_idx = 0;
        lvl_regi_coarse.init_mov_vol_poses = { init_guess };
        lvl_regi_coarse.ref_frames = { 0 };
        {
          auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
          cmaes_regi->set_opt_vars(se3_vars);
          cmaes_regi->set_opt_x_tol(0.01);
          cmaes_regi->set_opt_obj_fn_tol(0.01);

          auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

          pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };
          pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(20, 20) };

          cmaes_regi->set_pop_size(50);
          cmaes_regi->set_sigma({ 30 * kDEG2RAD, 30 * kDEG2RAD, 30 * kDEG2RAD, 50, 50, 100 });

          cmaes_regi->set_penalty_fn(pen_fn);
          cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

          lvl_regi_coarse.regi = cmaes_regi;
        }
      }

      if (kSAVE_REGI_DEBUG )
      {
        // Create Debug Proj H5 File
        const std::string proj_data_h5_path = output_path + "/spine_singleview_proj_data" + exp_ID + ".h5";
        vout << "creating H5 proj data file for img" + exp_ID + "..." << std::endl;
        H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
        WriteProjDataH5(regi.fixed_proj_data, &h5);

        vout << "  setting regi debug info..." << std::endl;

        DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
        debug_vol_path.vol_path = resample_vert_vol_path;

        if (false)
        {
          debug_vol_path.label_vol_path = spineseg_path;
          debug_vol_path.labels_used    = { spine_label };
        }

        regi.debug_info->vols = { debug_vol_path };

        DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
        debug_proj_path.path = proj_data_h5_path;
        // debug_proj_path.projs_used = { view_idx };

        regi.debug_info->fixed_projs = debug_proj_path;

        regi.debug_info->regi_names = { { "Singleview Spine" + exp_ID } };
      }
    }

    vout << std::endl << "Single view spine registration ..." << std::endl;
    regi.run();
    regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    const FrameTransform spine_regi_xform = regi.cur_cam_to_vols[0];

    if (kSAVE_REGI_DEBUG)
    {
      vout << "writing debug info to disk..." << std::endl;
      const std::string dst_debug_path = output_path + "/debug_spine" + exp_ID + ".h5";
      WriteMultiLevel2D3DRegiDebugToDisk(*regi.debug_info, dst_debug_path);
      WriteITKAffineTransform(output_path + "/spine_regi_xform" + exp_ID + ".h5", spine_regi_xform);
    }

    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_model(default_cam);
      ray_caster->use_proj_store_replace_method();
      ray_caster->set_volume(resample_vert);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();
      ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[0];
      ray_caster->compute(0);

      if (kSAVE_REGI_DEBUG)
        WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/spine_reproj" + exp_ID + ".png");

      H5::Group dst_spine_reproj_g = dst_h5.createGroup("reproj-spine-img");
      WriteImageH5(ray_caster->proj(0).GetPointer(), &dst_spine_reproj_g);
    }


    regi.fixed_proj_data = proj_device_list;
    regi.levels.resize(1);
    // Device Single-view Registration
    {
      auto& lvl = regi.levels[0];

      lvl.ds_factor = 0.25;

      lvl.fixed_imgs_to_use = { view_idx };

      lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

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
        auto& lvl_regi_coarse = lvl.regis[0];
        lvl_regi_coarse.mov_vols = { 1 };
        lvl_regi_coarse.static_vols = { };

        auto init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        init_guess->vol_idx = 1;
        lvl_regi_coarse.init_mov_vol_poses = { init_guess };
        lvl_regi_coarse.ref_frames = { 1 };
        {
          auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
          cmaes_regi->set_opt_vars(se3_vars);
          cmaes_regi->set_opt_x_tol(0.01);
          cmaes_regi->set_opt_obj_fn_tol(0.01);

          auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

          pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };
          pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(20, 20) };

          cmaes_regi->set_pop_size(50);
          cmaes_regi->set_sigma({ 30 * kDEG2RAD, 30 * kDEG2RAD, 30 * kDEG2RAD, 50, 50, 100 });

          cmaes_regi->set_penalty_fn(pen_fn);
          cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

          lvl_regi_coarse.regi = cmaes_regi;
        }
      }

      if (kSAVE_REGI_DEBUG )
      {
        // Create Debug Proj H5 File
        const std::string proj_data_h5_path = output_path + "/device_singleview_proj_data" + exp_ID + ".h5";
        vout << "creating H5 proj data file for img" + exp_ID + "..." << std::endl;
        H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
        WriteProjDataH5(regi.fixed_proj_data, &h5);

        vout << "  setting regi debug info..." << std::endl;

        DebugRegiResultsMultiLevel::VolPathInfo debug_spinevol_path;
        DebugRegiResultsMultiLevel::VolPathInfo debug_devicevol_path;
        debug_spinevol_path.vol_path = spinevol_path;
        debug_devicevol_path.vol_path = devicevol_path;

        if (use_seg)
        {
          debug_spinevol_path.label_vol_path = { spineseg_path };
          debug_spinevol_path.labels_used    = { spine_label };
          debug_devicevol_path.label_vol_path = { deviceseg_path };
          debug_devicevol_path.labels_used    = { device_label };
        }

        regi.debug_info->vols = { debug_spinevol_path, debug_devicevol_path };

        DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
        debug_proj_path.path = proj_data_h5_path;
        // debug_proj_path.projs_used = { view_idx };

        regi.debug_info->fixed_projs = debug_proj_path;

        regi.debug_info->regi_names = { { "Singleview Device" + exp_ID } };
      }
    }

    vout << std::endl << "Single view device registration ..." << std::endl;
    regi.run();
    regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    const FrameTransform device_regi_xform = regi.cur_cam_to_vols[1];

    if (kSAVE_REGI_DEBUG)
    {
      vout << "writing debug info to disk..." << std::endl;
      const std::string dst_debug_path = output_path + "/debug_device" + exp_ID + ".h5";
      WriteMultiLevel2D3DRegiDebugToDisk(*regi.debug_info, dst_debug_path);
      WriteITKAffineTransform(output_path + "/device_regi_xform" + exp_ID + ".h5", device_regi_xform);
    }

    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_model(default_cam);
      ray_caster->use_proj_store_replace_method();
      ray_caster->set_volume(device_vol);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();
      ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[1];
      ray_caster->compute(0);

      if (kSAVE_REGI_DEBUG)
        WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/device_reproj" + exp_ID + ".png");

      H5::Group dst_device_reproj_g = dst_h5.createGroup("reproj-device-img");
      WriteImageH5(ray_caster->proj(0).GetPointer(), &dst_device_reproj_g);
    }

    // Log Regi Result Files
    {
      WriteAffineTransform4x4("gt-cam-wrt-spine", gt_cam_wrt_spine, &dst_h5);
      WriteAffineTransform4x4("gt-cam-wrt-device", gt_cam_wrt_device, &dst_h5);
      WriteAffineTransform4x4("init-cam-wrt-spine", init_cam_wrt_spine, &dst_h5);
      WriteAffineTransform4x4("init-cam-wrt-device", init_cam_wrt_device, &dst_h5);
      WriteAffineTransform4x4("regi-cam-wrt-spine", spine_regi_xform, &dst_h5);
      WriteAffineTransform4x4("regi-cam-wrt-device", device_regi_xform, &dst_h5);
      WriteSingleScalarH5("vert-rotX", rot_angX_rand, &dst_h5);
      WriteSingleScalarH5("vert-rotY", rot_angY_rand, &dst_h5);
      WriteSingleScalarH5("vert-rotZ", rot_angZ_rand, &dst_h5);
      dst_h5.close();
    }
  }
  return 0;
}
