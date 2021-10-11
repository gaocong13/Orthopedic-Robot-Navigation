
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
#include "xregImageAddPoissonNoise.h"
#include "xregSampleUtils.h"

#include "xregPAODrawBones.h"
#include "xregRigidUtils.h"

#include "bigssMath.h"

using namespace xreg;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;
constexpr bool kSAVE_REGI_DEBUG = true;

using size_type = std::size_t;
using CamModelList = std::vector<CameraModel>;

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

  po.set_help("Multi-view Multi-Configuration Device Registration");
  po.set_arg_usage("<Meta Data Path> <Output folder path>");
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

  const std::string meta_data_path = po.pos_args()[0];  // 3D device landmarks path
  const std::string output_path    = po.pos_args()[1];  // Output path

  const size_type num_views = 3;
  const size_type num_config = 3;

  const std::string UR_kins_path  = meta_data_path + "/Ukins";
  const std::string exp_list_path = meta_data_path + "/expID.txt";  // Experiment list file path

  const std::string spinevol_path = meta_data_path + "/Spine21-2512_CT_crop.nrrd";
  const std::string spineseg_path = meta_data_path + "/Sheetness_seg_crop_mapped.nrrd";
  const std::string spine_gt_xform_path = meta_data_path + "/sacrum_regi_xform.h5";
  const std::string spine_3d_fcsv_path = meta_data_path + "/Spine_3D_landmarks.fcsv";

  const std::string handeye_X_path = meta_data_path + "/devicehandeye_X.h5";
  const std::string device_3d_fcsv_path = meta_data_path + "/Device3Dlandmark.fcsv";
  const std::string device_3d_bb_fcsv_path = meta_data_path + "/Device3Dbb.fcsv";
  const std::string devicevol_path = meta_data_path + "/Device_crop_CT.nii.gz";
  const std::string deviceseg_path = meta_data_path + "/Device_crop_seg.nii.gz";

  const std::string device_gt_xform_view0_config1_path = meta_data_path + "/device_regi_xform01.h5";
  const std::string device_gt_xform_view0_config2_path = meta_data_path + "/device_regi_xform02.h5";
  const std::string device_gt_xform_view0_config3_path = meta_data_path + "/device_regi_xform03.h5";
  const std::string device_gt_xform_view1_config1_path = meta_data_path + "/device_regi_xform05.h5";
  const std::string device_gt_xform_view1_config2_path = meta_data_path + "/device_regi_xform06.h5";
  const std::string device_gt_xform_view1_config3_path = meta_data_path + "/device_regi_xform07.h5";
  const std::string device_gt_xform_view2_config1_path = meta_data_path + "/device_regi_xform09.h5";
  const std::string device_gt_xform_view2_config2_path = meta_data_path + "/device_regi_xform10.h5";
  const std::string device_gt_xform_view2_config3_path = meta_data_path + "/device_regi_xform11.h5";

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

  if(lineNumber!=exp_ID_list.size()) throw std::runtime_error("Exp ID list size mismatch!!!");

  std::cout << "reading spine anatomical landmarks from FCSV file..." << std::endl;
  auto spine_3d_fcsv = ReadFCSVFileNamePtMap(spine_3d_fcsv_path);
  ConvertRASToLPS(&spine_3d_fcsv);

  std::cout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_fcsv = ReadFCSVFileNamePtMap(device_3d_fcsv_path);
  ConvertRASToLPS(&device_3d_fcsv);

  const bool use_seg = true;
  auto spine_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(spineseg_path);

  vout << "reading spine volume..." << std::endl; // We only use the needle metal part
  auto spinevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(spinevol_path);

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

  // TODO: check handeye xform
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

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> spine_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> spine_multiview_regi_ref_frame;
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

    FrameTransform spine_vol_to_centered_vol = FrameTransform::Identity();
    spine_vol_to_centered_vol.matrix()(0,3) = -spine_rotcenter[0];
    spine_vol_to_centered_vol.matrix()(1,3) = -spine_rotcenter[1];
    spine_vol_to_centered_vol.matrix()(2,3) = -spine_rotcenter[2];

    spine_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(spine_vol_to_centered_vol, true);
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> device_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> device_multiview_regi_ref_frame;
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

    FrameTransform device_vol_to_centered_vol = FrameTransform::Identity();
    device_vol_to_centered_vol.matrix()(0,3) = -device_rotcenter[0];
    device_vol_to_centered_vol.matrix()(1,3) = -device_rotcenter[1];
    device_vol_to_centered_vol.matrix()(2,3) = -device_rotcenter[2];

    device_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(device_vol_to_centered_vol, true);
  }

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  const FrameTransform gt_cam_wrt_spine = ReadITKAffineTransformFromFile(spine_gt_xform_path);
  // TODO: Modify the device_gt_xform_path
  const FrameTransform gt_cam_wrt_device_view0_config1 = ReadITKAffineTransformFromFile(device_gt_xform_view0_config1_path);
  const FrameTransform gt_cam_wrt_device_view0_config2 = ReadITKAffineTransformFromFile(device_gt_xform_view0_config2_path);
  const FrameTransform gt_cam_wrt_device_view0_config3 = ReadITKAffineTransformFromFile(device_gt_xform_view0_config3_path);
  const FrameTransform gt_cam_wrt_device_view1_config1 = ReadITKAffineTransformFromFile(device_gt_xform_view1_config1_path);
  const FrameTransform gt_cam_wrt_device_view1_config2 = ReadITKAffineTransformFromFile(device_gt_xform_view1_config2_path);
  const FrameTransform gt_cam_wrt_device_view1_config3 = ReadITKAffineTransformFromFile(device_gt_xform_view1_config3_path);
  const FrameTransform gt_cam_wrt_device_view2_config1 = ReadITKAffineTransformFromFile(device_gt_xform_view2_config1_path);
  const FrameTransform gt_cam_wrt_device_view2_config2 = ReadITKAffineTransformFromFile(device_gt_xform_view2_config2_path);
  const FrameTransform gt_cam_wrt_device_view2_config3 = ReadITKAffineTransformFromFile(device_gt_xform_view2_config3_path);

  FrameTransformList gt_cam_wrt_device_list = { gt_cam_wrt_device_view0_config1, gt_cam_wrt_device_view0_config2, gt_cam_wrt_device_view0_config3,
                                                gt_cam_wrt_device_view1_config1, gt_cam_wrt_device_view1_config2, gt_cam_wrt_device_view1_config3,
                                                gt_cam_wrt_device_view2_config1, gt_cam_wrt_device_view2_config2, gt_cam_wrt_device_view2_config3 };
  FrameTransformList gt_cam_wrt_spine_list;
  gt_cam_wrt_spine_list.reserve(num_views);
  for (size_type idx = 0; idx < num_views; ++idx)
  {
      gt_cam_wrt_spine_list[idx] = gt_cam_wrt_spine * gt_cam_wrt_device_list[0].inverse() * gt_cam_wrt_device_list[idx*num_config];
  }

  ProjDataF32List proj_spine_mv_list;
  ProjDataF32List proj_device_mv_list;
  proj_spine_mv_list.reserve(num_views);
  proj_device_mv_list.reserve(num_views);

  std::vector<ProjDataF32List> proj_device_mv_mc_list;

  ProjDataF32 proj_spine;
  ProjDataF32 proj_device;

  CamModelList sim_cam;
  sim_cam = { default_cam };

  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    for (size_type device_flag = 0; device_flag < 2; device_flag++)
    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_models(sim_cam);
      ray_caster->set_volumes({devicevol_att, spinevol_att});
      // ray_caster->set_ray_step_size(0.5);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();

      if (device_flag == 1)
      {
        ray_caster->use_proj_store_replace_method();
        // TODO: change gt pose
        ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_device_list[view_idx*num_config]);
        ray_caster->compute(0);
        ray_caster->use_proj_store_accum_method();
      }
      // TODO: change gt pose
      ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_spine_list[view_idx]);
      ray_caster->compute(1);

      vout << "projecting view: " << view_idx << " device flag: " << device_flag << "..." << std::endl;

      if (device_flag == 1)
      {
        proj_device.img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 2000);
        proj_device.cam = default_cam;
        proj_device_mv_list.push_back( proj_device );
        WriteITKImageRemap8bpp(proj_device.img.GetPointer(), output_path + "/device_mv_view" + std::to_string(view_idx) + ".png");
      }
      else
      {
        proj_spine.img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 2000);
        proj_spine.cam = default_cam;
        proj_spine_mv_list.push_back( proj_spine );
        WriteITKImageRemap8bpp(proj_spine.img.GetPointer(), output_path + "/spine_mv_view" + std::to_string(view_idx) + ".png");
      }
    }

    ProjDataF32List proj_device_mc_list;
    for (size_type config_idx = 0; config_idx < num_config; config_idx++)
    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_models(sim_cam);
      ray_caster->set_volumes({devicevol_att, spinevol_att});
      // ray_caster->set_ray_step_size(0.5);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();

      {
        ray_caster->use_proj_store_replace_method();
        // TODO: change gt pose
        ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_device_list[view_idx*num_config+config_idx]);
        ray_caster->compute(0);
        ray_caster->use_proj_store_accum_method();
      }
      // TODO: change gt pose
      ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_spine_list[view_idx]);
      ray_caster->compute(1);

      {
        proj_device.img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 2000);
        proj_device.cam = default_cam;
        proj_device_mc_list.push_back( proj_device );
        WriteITKImageRemap8bpp(proj_device.img.GetPointer(), output_path + "/device_mc_view" + std::to_string(view_idx) + "config" + std::to_string(config_idx) + ".png");
      }
    }

    proj_device_mv_mc_list.push_back(proj_device_mc_list);
  }

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

  // Create 3D pnp landmark map
  for(size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    vout << "Reading URkins..." << std::endl;
    FrameTransformList UReef_xform_list;
    for(size_type config_idx = 0; config_idx < num_config; ++config_idx)
    {
      FrameTransform regi_UReef_xform;
      {
        const std::string src_ureef_path          = UR_kins_path + "/" + exp_ID_list[view_idx*num_views + config_idx] + "/ur_eef.h5";
        H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
        H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
        H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
        std::vector<float> UReef_tracker          = ReadVectorH5Float("TranformParameters", ureef_group0);
        regi_UReef_xform                         = ConvertSlicerToITK(UReef_tracker);
      }
      UReef_xform_list.push_back(regi_UReef_xform);
    }

    LandMap3 pnp_vol_lands;
    LandMap2 pnp_img_lands;

    for(size_type config_idx = 0; config_idx < num_config; ++config_idx)
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

        Pt3 cur_ld = default_cam.phys_pt_to_ind_pt(Pt3(gt_cam_wrt_device_list[view_idx*num_config+config_idx].inverse() * n.second));

        int cur_ld_x = static_cast<int>(cur_ld[0] + 0.5);
        int cur_ld_y = static_cast<int>(cur_ld[1] + 0.5);

        // Insert to 2D landmark map
        std::mt19937 rng_eng;
        SeedRNGEngWithRandDev(&rng_eng);

        // Add some noise for 2D landmark positions
        std::uniform_real_distribution<CoordScalar> land2d_noise_dist(-3, 3);

        // Add 2D landmark to pnp_img_lands list
        Pt2 cur_ld_2D_pt = {cur_ld_x, cur_ld_y};
        std::pair<std::string, Pt2> ld_2D(std::to_string(config_idx) + n.first, cur_ld_2D_pt);
        pnp_img_lands.insert(ld_2D);
      }
    }

    // Define a refernece frame at the center of ld1_3D_pts
    FrameTransform initref_xform = FrameTransform::Identity();
    initref_xform(0,3) = -(initref_minX + initref_maxX)/2;
    initref_xform(1,3) = -(initref_minY + initref_maxY)/2;
    initref_xform(2,3) = -(initref_minZ + initref_maxZ)/2;

    for(size_type config_idx=0; config_idx < num_config; ++config_idx)
    {
      for( const auto& n : device_3d_fcsv )
      {
        auto ld_3D_pt_init = handeye_ref_xform.inverse() * handeye_regi_X.inverse() * UReef_xform_list[0].inverse() * UReef_xform_list[view_idx] * handeye_regi_X * handeye_ref_xform * n.second;
        auto ld_3D_pt_ref = initref_xform * ld_3D_pt_init;
        std::pair<std::string, Pt3> ld_3D(std::to_string(view_idx) + n.first, ld_3D_pt_ref);
        pnp_vol_lands.insert(ld_3D);
      }
    }

    vout << "Calculating PnP initial pose..." << std::endl;
    const FrameTransform lands_cam_to_initref = PnPPOSITAndReprojCMAES(default_cam, pnp_vol_lands, pnp_img_lands);

    // ################### Start Multi-Config Registration ##############################

    MultiLevelMultiObjRegi regi_mc; // multi-config registration

    regi_mc.set_debug_output_stream(vout, verbose);
    regi_mc.set_save_debug_info(kSAVE_REGI_DEBUG);

    for (size_type config_idx = 0; config_idx < num_config; ++config_idx)
    {
      regi_mc.vols.push_back( device_vol );
      regi_mc.vol_names.push_back( "Device" );
    }

    regi_mc.ref_frames = { device_singleview_regi_ref_frame };

    regi_mc.init_cam_to_vols = { };

    vout << "Setting up registration initial poses..." << std::endl;
    for(size_type config_idx = 0; config_idx < num_config; ++config_idx)
    {
      FrameTransform init_cam_wrt_device = handeye_ref_xform.inverse() * handeye_regi_X.inverse() * UReef_xform_list[view_idx].inverse()
                              * UReef_xform_list[0] * handeye_regi_X * handeye_ref_xform * initref_xform.inverse() * lands_cam_to_initref;

      regi_mc.init_cam_to_vols.push_back( init_cam_wrt_device );

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

        WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/deice_init_reproj" + exp_ID_list[config_idx] + ".png");
      }
    }

    regi_mc.fixed_proj_data = proj_device_mv_mc_list[view_idx];

    device_singleview_regi_ref_frame->cam_extrins = regi_mc.fixed_proj_data[0].cam.extrins;

    auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();

    regi_mc.levels.resize(1);

    auto& lvl = regi_mc.levels[0];

    lvl.ds_factor = 0.25;

    lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

    lvl.fixed_imgs_to_use.resize(num_config);
    std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

    vout << "Setting up similarity metrics..." << std::endl;
    for (size_type config_idx = 0; config_idx < num_config; ++config_idx)
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
      auto& lvl_regi = lvl.regis[0];
      lvl_regi.mov_vols = { };
      lvl_regi.init_mov_vol_poses = {};
      lvl_regi.ref_frames = { };
      for(size_type config_idx = 0; config_idx < num_config; ++config_idx)
      {
        lvl_regi.mov_vols.push_back( config_idx );

        auto init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        init_guess->vol_idx = config_idx;
        lvl_regi.init_mov_vol_poses.push_back( init_guess );
        lvl_regi.ref_frames.push_back( 0 );
      }
      lvl_regi.static_vols = { };

      {
        auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAESdeviceMultiConfig>();

        cmaes_regi->set_opt_vars(se3_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        cmaes_regi->set_UReef_xform_list(UReef_xform_list);
        cmaes_regi->set_initref_xform(initref_xform);
        cmaes_regi->set_handeye_ref_xform(handeye_ref_xform);
        cmaes_regi->set_handeyeX(handeye_regi_X);

        auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

        pen_fn->rot_pdfs_per_obj   = { };
        pen_fn->trans_pdfs_per_obj = { };
        for(size_type config_idx = 0; config_idx < num_config; ++config_idx)
        {
          pen_fn->rot_pdfs_per_obj.push_back( std::make_shared<FoldNormDist>(2.5 * kDEG2RAD, 2.5 * kDEG2RAD) );
          pen_fn->trans_pdfs_per_obj.push_back( std::make_shared<FoldNormDist>(5, 5) );
        }

        cmaes_regi->set_pop_size(20);
        cmaes_regi->set_sigma({ 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5, 2.5, 25 });

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

        lvl_regi.regi = cmaes_regi;
      }
    }

    if (kSAVE_REGI_DEBUG )
    {
      // Create Debug Proj H5 File
      const std::string proj_data_h5_path = output_path + "/spine_multiconfig_proj_data_view" + fmt::format("{:02d}", view_idx) + ".h5";
      vout << "creating H5 proj data file for view" + fmt::format("{:02d}", view_idx) + "..." << std::endl;
      H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
      WriteProjDataH5(regi_mc.fixed_proj_data, &h5);

      vout << "  setting regi debug info..." << std::endl;

      DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
      debug_vol_path.vol_path = devicevol_path;

      if (use_seg)
      {
        debug_vol_path.label_vol_path = deviceseg_path;
        debug_vol_path.labels_used    = { };
        for(size_type config_idx = 0; config_idx < num_config; ++config_idx)
        {
          debug_vol_path.labels_used.push_back( device_label );
        }
      }

      regi_mc.debug_info->vols = {  };
      for(size_type config_idx = 0; config_idx < num_config; ++config_idx)
      {
        regi_mc.debug_info->vols.push_back( debug_vol_path );
      }

      DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
      debug_proj_path.path = proj_data_h5_path;
      // debug_proj_path.projs_used = { view_idx };

      regi_mc.debug_info->fixed_projs = debug_proj_path;

      regi_mc.debug_info->regi_names = { { "Multiconfig DeviceRegi" + fmt::format("{:02d}", view_idx) } };
    }

    vout << std::endl << "Running registration ..." << std::endl;
    regi_mc.run();
    regi_mc.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    if (kSAVE_REGI_DEBUG)
    {
      vout << "writing debug info to disk..." << std::endl;
      const std::string dst_debug_path = output_path + "/debug_device" + fmt::format("{:02d}", view_idx) + ".h5";
      WriteMultiLevel2D3DRegiDebugToDisk(*regi_mc.debug_info, dst_debug_path);
    }
  }

  return 0;
}
