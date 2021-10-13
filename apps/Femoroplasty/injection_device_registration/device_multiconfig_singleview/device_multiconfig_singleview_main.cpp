
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
#include "xregHDF5.h"
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
#include "xregIntensity2D3DRegiCMAES-Device-MultiConfig.h"
#include "xregIntensity2D3DRegiBOBYQA.h"
#include "xregRegi2D3DPenaltyFnSE3Mag.h"
#include "xregFoldNormDist.h"
#include "xregHipSegUtils.h"

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

  po.set_help("Single-view Device Registration");
  po.set_arg_usage("<Device 2D landmark annotation ROOT path> <Device 3D landmark annotation FILE path> <Device 3D BB FILE path> <Image ID list txt file path> <Image DICOM ROOT path> <Output folder path>");
  po.set_min_num_pos_args(6);

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

  const std::string device_2d_fcsv_root_path  = po.pos_args()[0];  // 2D Landmark root path
  const std::string device_3d_fcsv_path       = po.pos_args()[1];  // 3D device landmarks path
  const std::string UR_kins_path              = po.pos_args()[2];  // UR kinematics path
  const std::string dicom_path                = po.pos_args()[3];  // Dicom image path
  const std::string output_path               = po.pos_args()[4];  // Output path
  const std::string exp_list_path             = po.pos_args()[5];  // Experiment list file path

  std::cout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_fcsv = ReadFCSVFileNamePtMap(device_3d_fcsv_path);
  ConvertRASToLPS(&device_3d_fcsv);

  const std::string handeye_X_path = "/home/cong/Research/Spine/CadaverNeedleInjection/Device_crop.nii.gz";
  const std::string devicevol_path = "/home/cong/Research/Spine/CadaverNeedleInjection/Device_crop.nii.gz";
  const std::string deviceseg_path = "/home/cong/Research/Spine/CadaverNeedleInjection/Device_crop_seg.nii.gz";

  FrameTransform device_rotcen_ref = FrameTransform::Identity();
  {
    auto deviceref_fcsv = device_3d_fcsv.find("RotCenter");
    Pt3 device_rotcen_pt;

    if (deviceref_fcsv != device_3d_fcsv.end()){
      device_rotcen_pt = deviceref_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND DEVICE ROT CENTER PT" << std::endl;
    }

    device_rotcen_ref.matrix()(0,3) = -device_rotcen_pt[0];
    device_rotcen_ref.matrix()(1,3) = -device_rotcen_pt[1];
    device_rotcen_ref.matrix()(2,3) = -device_rotcen_pt[2];
  }

  FrameTransform handeye_ref_xform = FrameTransform::Identity();
  {
    auto handeye_ref_fcsv = device_3d_fcsv.find("RotCenter");
    Pt3 handeye_ref_pt;

    if (handeye_ref_fcsv != device_3d_fcsv.end()){
      handeye_ref_pt = handeye_ref_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND DEVICE ROT CENTER PT" << std::endl;
    }

    handeye_ref_xform.matrix()(0,3) = -handeye_ref_pt[0];
    handeye_ref_xform.matrix()(1,3) = -handeye_ref_pt[1];
    handeye_ref_xform.matrix()(2,3) = -handeye_ref_pt[2];
  }

  const bool use_seg = true;
  auto device_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(deviceseg_path);

  vout << "reading device volume..." << std::endl; // We only use the needle metal part
  auto devicevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(devicevol_path);

  vout << "  HU --> Att. ..." << std::endl;

  auto devicevol_att = HUToLinAtt(devicevol_hu.GetPointer());

  unsigned char device_label = 1;

  auto device_vol = ApplyMaskToITKImage(devicevol_att.GetPointer(), device_seg.GetPointer(), device_label, float(0), true);

  std::vector<std::string> exp_ID_list;
  size_type lineNumber = 0;
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

  const size_type num_views = lineNumber;

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> device_singleview_regi_ref_frame;

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
    device_singleview_regi_ref_frame->vol_idx = 0;
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = device_rotcenter[0];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = device_rotcenter[1];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = device_rotcenter[2];
  }

  if(lineNumber!=exp_ID_list.size()) throw std::runtime_error("Exp ID list size mismatch!!!");

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  ProjPreProc proj_pre_proc;
  proj_pre_proc.input_projs.resize(num_views);

  std::vector<CIOSFusionDICOMInfo> devicecios_metas(num_views);
  for(size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    const std::string img_path = dicom_path + "/" + exp_ID_list[view_idx];
    std::tie(proj_pre_proc.input_projs[0].img, devicecios_metas[0]) = ReadCIOSFusionDICOMFloat(img_path);
    proj_pre_proc.input_projs[0].cam = NaiveCamModelFromCIOSFusion(devicecios_metas[0], true);
  }

  proj_pre_proc();

  auto& projs_to_regi = proj_pre_proc.output_projs;

  // ###################### Multi-configuration initialization ###############################
  Pt3 ld_3D_pt_init, ld2_3D_pt_init;
  std::vector<float> initref_vecX, initref_vecY, initref_vecZ;
  float initref_minX = 100000;
  float initref_maxX = -100000;
  float initref_minY = 100000;
  float initref_maxY = -100000;
  float initref_minZ = 100000;
  float initref_maxZ = -100000;
  std::vector<Pt3> ld_3D_pt_init_list;
  FrameTransform handeye_regi_X = ReadITKAffineTransformFromFile(handeye_X_path);

  FrameTransformList UReef_xform_list;
  for(size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    FrameTransform regi_UReef_xform;
    {
      const std::string src_ureef_path          = UR_kins_path + "/" + exp_ID_list[view_idx] + "/ur_eef.h5";
      H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
      H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
      H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
      std::vector<float> UReef_tracker          = ReadVectorH5Float("TranformParameters", ureef_group0);
      regi_UReef_xform                         = ConvertSlicerToITK(UReef_tracker);
    }
    UReef_xform_list.push_back(regi_UReef_xform);
  }

  // Create 3D pnp landmark map
  LandMap3 pnp_vol_lands;
  LandMap2 pnp_img_lands;
  for(size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    // Assuming first view device volume frame as reference frame;
    // ld_3D_pt_init is i-th view base landmark w.r.t. first view base seg volume frame
    for( const auto& n : device_3d_fcsv )
    {
      auto ld_3D_pt_init = handeye_ref_xform.inverse() * handeye_regi_X.inverse() * UReef_xform_list[0].inverse() * UReef_xform_list[view_idx] * handeye_regi_X * handeye_ref_xform * n.second;
      ld_3D_pt_init_list.push_back(ld_3D_pt_init);

      // Sum over all the points w.r.t. first view device volume frame
      initref_vecX.push_back( ld_3D_pt_init(0) );
      initref_vecY.push_back( ld_3D_pt_init(1) );
      initref_vecZ.push_back( ld_3D_pt_init(2) );

      if( ld_3D_pt_init(0) > initref_maxX )
        initref_maxX = ld_3D_pt_init(0);

      if( ld_3D_pt_init(1) > initref_maxY )
        initref_maxY = ld_3D_pt_init(1);

      if( ld_3D_pt_init(2) > initref_maxZ )
        initref_maxZ = ld_3D_pt_init(2);

      if( ld_3D_pt_init(0) < initref_minX )
        initref_minX = ld_3D_pt_init(0);

      if( ld_3D_pt_init(1) < initref_minY )
        initref_minY = ld_3D_pt_init(1);

      if( ld_3D_pt_init(2) < initref_minZ )
        initref_minZ = ld_3D_pt_init(2);
    }

    const std::string device_2d_fcsv_path = device_2d_fcsv_root_path + "/" + exp_ID_list[view_idx] + ".fcsv";
    auto device_2d_fcsv = ReadFCSVFileNamePtMap(device_2d_fcsv_path);
    ConvertRASToLPS(&device_2d_fcsv);

    for( const auto& n : device_2d_fcsv )
    {
      std::pair<std::string, Pt2> pnp_ld(std::to_string(view_idx) + n.first, Pt2{(1536 - n.second[0]/0.194), (1536 - n.second[1]/0.194)});
      pnp_img_lands.insert(pnp_ld);
    }
  }

  // Average over all points

  // Define a refernece frame at the center of ld1_3D_pts
  FrameTransform initref_xform = FrameTransform::Identity();
  initref_xform(0,3) = -(initref_minX + initref_maxX)/2;
  initref_xform(1,3) = -(initref_minY + initref_maxY)/2;
  initref_xform(2,3) = -(initref_minZ + initref_maxZ)/2;

  for(size_type view_idx=0; view_idx < num_views; ++view_idx)
  {
    for( const auto& n : device_3d_fcsv )
    {
      auto ld_3D_pt_init = handeye_ref_xform.inverse() * handeye_regi_X.inverse() * UReef_xform_list[0].inverse() * UReef_xform_list[view_idx] * handeye_regi_X * handeye_ref_xform * n.second;
      auto ld_3D_pt_ref = initref_xform * ld_3D_pt_init;
      std::pair<std::string, Pt3> ld_3D(std::to_string(view_idx) + n.first, ld_3D_pt_ref);
      pnp_vol_lands.insert(ld_3D);
    }
  }

  const FrameTransform lands_cam_to_initref = PnPPOSITAndReprojCMAES(default_cam, pnp_vol_lands, pnp_img_lands);

  bool is_first_view = true;

  MultiLevelMultiObjRegi regi;

  regi.set_debug_output_stream(vout, verbose);
  regi.set_save_debug_info(kSAVE_REGI_DEBUG);

  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    regi.vols.push_back( device_vol );
    regi.vol_names.push_back( "Device" );
  }

  regi.ref_frames = { device_singleview_regi_ref_frame };

  regi.init_cam_to_vols = { };

  for(size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    FrameTransform init_cam_wrt_device = handeye_ref_xform.inverse() * handeye_regi_X.inverse() * UReef_xform_list[view_idx].inverse()
                            * UReef_xform_list[0] * handeye_regi_X * handeye_ref_xform * initref_xform.inverse() * lands_cam_to_initref;

    regi.init_cam_to_vols.push_back( init_cam_wrt_device );

    // ###################### Reprojection for initial pnp poses ###############################
    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_model(default_cam);
      ray_caster->use_proj_store_replace_method();
      ray_caster->set_volume(device_vol);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();
      ray_caster->xform_cam_to_itk_phys(0) = init_cam_wrt_device;
      ray_caster->compute(0);

      WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/deice_init_reproj" + exp_ID_list[view_idx] + ".png");
    }
  }

  device_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[0].cam.extrins;

  auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();

  const size_type view_idx = 0;

  regi.fixed_proj_data = proj_pre_proc.output_projs;

  regi.levels.resize(1);

  auto& lvl = regi.levels[0];

  lvl.ds_factor = 0.25;

  lvl.fixed_imgs_to_use = { view_idx };

  lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

  lvl.fixed_imgs_to_use.resize(num_views);
  std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

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
    auto& lvl_regi_coarse = lvl.regis[0];
    lvl_regi_coarse.mov_vols = { };
    lvl_regi_coarse.init_mov_vol_poses = {};
    lvl_regi_coarse.ref_frames = { };
    for(size_type view_idx = 0; view_idx < num_views; ++view_idx)
    {
      lvl_regi_coarse.mov_vols.push_back( view_idx );

      auto init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      init_guess->vol_idx = view_idx;
      lvl_regi_coarse.init_mov_vol_poses.push_back( init_guess );
      lvl_regi_coarse.ref_frames.push_back( 0 );
    }
    lvl_regi_coarse.static_vols = { };

    {
      auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAESdeviceMultiConfig>();
      /*
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(0.01);
      cmaes_regi->set_opt_obj_fn_tol(0.01);

      cmaes_regi->set_UReef_xform_list(UReef_xform_list);
      cmaes_regi->set_initref_xform(initref_xform);
      cmaes_regi->set_handeye_ref_xform(handeye_ref_xform);
      cmaes_regi->set_handeyeX(handeye_regi_X);

      auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

      pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(2.5 * kDEG2RAD, 2.5 * kDEG2RAD) };
      pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(5, 5) };

      cmaes_regi->set_pop_size(20);
      cmaes_regi->set_sigma({ 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5, 2.5, 25 });

      cmaes_regi->set_penalty_fn(pen_fn);
      cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);
      */

      lvl_regi_coarse.regi = cmaes_regi;
    }
    vout << std::endl << "First view spine coarse registration ..." << std::endl;
    regi.run();
    regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

  }

  const std::string device_3d_bb_fcsv_path = "/Users/gaocong/Documents/Research/Femoroplasty/Phantom_Drilling/meta_data/Device3Dbb.fcsv";

  std::cout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_bb_fcsv = ReadFCSVFileNamePtMap(device_3d_bb_fcsv_path);
  ConvertRASToLPS(&device_3d_bb_fcsv);

  for(size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    WriteITKAffineTransform(output_path + "/device_regi_xform" + exp_ID_list[view_idx] + ".h5", regi.cur_cam_to_vols[view_idx]);

    auto ray_caster = LineIntRayCasterFromProgOpts(po);
    ray_caster->set_camera_model(default_cam);
    ray_caster->use_proj_store_replace_method();
    ray_caster->set_volume(device_vol);
    ray_caster->set_num_projs(1);
    ray_caster->allocate_resources();
    ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[view_idx];
    ray_caster->compute(0);

    WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/device_reproj" + exp_ID_list[view_idx] + ".png");
    WriteITKImageRemap8bpp(proj_pre_proc.input_projs[view_idx].img.GetPointer(), output_path + "/real" + exp_ID_list[view_idx] + ".png");

    LandMap3 reproj_bbs_fcsv;

    for( const auto& n : device_3d_bb_fcsv )
    {
      auto reproj_bb = default_cam.phys_pt_to_ind_pt(Pt3(regi.cur_cam_to_vols[view_idx].inverse() *  n.second));
      reproj_bb[0] = 0.194 * (reproj_bb[0] - 1536);
      reproj_bb[1] = 0.194 * (reproj_bb[1] - 1536);
      reproj_bb[2] = 0;
      std::pair<std::string, Pt3> ld2_3D(n.first, reproj_bb);
      reproj_bbs_fcsv.insert(ld2_3D);
    }

    WriteFCSVFileFromNamePtMap(output_path + "/reproj_bb" + exp_ID_list[view_idx] + ".fcsv", reproj_bbs_fcsv);
  }

  // ###################### Check reprojection using kinematics ###############################
  size_type reproj_idx = 1;
  {
    FrameTransform reproj_device_xform = device_rotcen_ref.inverse() * handeye_regi_X.inverse() * UReef_xform_list[reproj_idx].inverse() * UReef_xform_list[0] * handeye_regi_X * device_rotcen_ref * regi.cur_cam_to_vols[0];

    auto ray_caster = LineIntRayCasterFromProgOpts(po);
    ray_caster->set_camera_model(default_cam);
    ray_caster->use_proj_store_replace_method();
    ray_caster->set_volume(device_vol);
    ray_caster->set_num_projs(1);
    ray_caster->allocate_resources();
    ray_caster->xform_cam_to_itk_phys(0) = reproj_device_xform;
    ray_caster->compute(0);

    WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/deice_URkins_repos" + exp_ID_list[reproj_idx] + ".png");
  }

  return 0;
}
