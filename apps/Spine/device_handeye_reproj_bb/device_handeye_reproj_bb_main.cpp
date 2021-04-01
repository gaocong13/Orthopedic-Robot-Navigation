
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

  po.set_help("Device pose calculation using PnP solution");
  po.set_arg_usage("<Device 2D landmark annotation ROOT path> <Device 3D landmark annotation FILE path> <Image ID list txt file path> <Image DICOM ROOT path> <Output folder path>");
  po.set_min_num_pos_args(5);

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

  const std::string meta_data_path            = po.pos_args()[0];  // 2D Landmark root path
  const std::string refdevice_xform_file_path = po.pos_args()[1];
  const std::string refUR_kins_path           = po.pos_args()[2];
  const std::string handeye_regi_file_path    = po.pos_args()[3];
  const std::string UR_kins_path              = po.pos_args()[4];
  const std::string exp_list_path             = po.pos_args()[5];  // Experiment list file path
  const std::string dicom_path                = po.pos_args()[6];  // Dicom image path
  const std::string output_path               = po.pos_args()[7];  // Output path

  const bool reproj_drr = false;

  const std::string device_3d_fcsv_path       = meta_data_path + "/Device3Dlandmark.fcsv";
  const std::string device_3d_bb_fcsv_path    = meta_data_path + "/Device3Dbb.fcsv";

  const std::string devicevol_path = meta_data_path + "/Device_crop_CT.nii.gz";
  const std::string deviceseg_path = meta_data_path + "/Device_crop_seg.nii.gz";
  auto device_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(deviceseg_path);

  vout << "reading device volume..." << std::endl; // We only use the needle metal part
  auto devicevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(devicevol_path);

  vout << "  HU --> Att. ..." << std::endl;

  auto devicevol_att = HUToLinAtt(devicevol_hu.GetPointer());

  unsigned char device_label = 1;

  auto device_vol = ApplyMaskToITKImage(devicevol_att.GetPointer(), device_seg.GetPointer(), device_label, float(0), true);

  FrameTransform refUReef_xform;
  {
    const std::string src_ureef_path          = refUR_kins_path;
    H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
    H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
    H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
    std::vector<float> UReef_tracker          = ReadVectorH5Float("TranformParameters", ureef_group0);
    refUReef_xform                            = ConvertSlicerToITK(UReef_tracker);
  }

  FrameTransform ref_device_xform = ReadITKAffineTransformFromFile(refdevice_xform_file_path);

  FrameTransform handeye_regi_X = ReadITKAffineTransformFromFile(handeye_regi_file_path);

  std::cout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_fcsv = ReadFCSVFileNamePtMap(device_3d_fcsv_path);
  ConvertRASToLPS(&device_3d_fcsv);

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

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  bool is_first_view = true;

  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string exp_ID                = exp_ID_list[idx];
    const std::string img_path              = dicom_path + "/" + exp_ID;

    std::cout << "Running..." << exp_ID << std::endl;

    ProjPreProc proj_pre_proc;
    proj_pre_proc.input_projs.resize(1);

    std::vector<CIOSFusionDICOMInfo> devicecios_metas(1);
    {
      std::tie(proj_pre_proc.input_projs[0].img, devicecios_metas[0]) =
                                                      ReadCIOSFusionDICOMFloat(img_path);
      proj_pre_proc.input_projs[0].cam = NaiveCamModelFromCIOSFusion(devicecios_metas[0], true);
    }

    proj_pre_proc();
    auto& projs_to_regi = proj_pre_proc.output_projs;

    FrameTransform init_cam_to_device;
    {
      const std::string src_ureef_path          = UR_kins_path + "/" + exp_ID_list[idx] + "/ur_eef.h5";
      H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
      H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
      H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
      std::vector<float> UReef_tracker          = ReadVectorH5Float("TranformParameters", ureef_group0);
      FrameTransform UReef_xform                = ConvertSlicerToITK(UReef_tracker);

      init_cam_to_device = device_rotcen_ref.inverse() * handeye_regi_X.inverse() * UReef_xform.inverse() * refUReef_xform * handeye_regi_X * device_rotcen_ref * ref_device_xform;
    }

    LandMap3 reproj_bbs_fcsv;

    std::cout << "reading device BB landmarks from FCSV file..." << std::endl;
    auto device_3d_bb_fcsv = ReadFCSVFileNamePtMap(device_3d_bb_fcsv_path);
    ConvertRASToLPS(&device_3d_bb_fcsv);

    for( const auto& n : device_3d_bb_fcsv )
    {
      auto reproj_bb = default_cam.phys_pt_to_ind_pt(Pt3(init_cam_to_device.inverse() *  n.second));
      reproj_bb[0] = 0.194 * (reproj_bb[0] - 1536);
      reproj_bb[1] = 0.194 * (reproj_bb[1] - 1536);
      reproj_bb[2] = 0;
      std::pair<std::string, Pt3> ld2_3D(n.first, reproj_bb);
      reproj_bbs_fcsv.insert(ld2_3D);
    }

    WriteFCSVFileFromNamePtMap(output_path + "/reproj_bb" + exp_ID + ".fcsv", reproj_bbs_fcsv);

    {
      LandMap3 reproj_screws_fcsv;

      for( const auto& n : device_3d_fcsv )
      {
        auto reproj_screws = default_cam.phys_pt_to_ind_pt(Pt3(init_cam_to_device.inverse() *  n.second));
        reproj_screws[0] = 0.194 * (reproj_screws[0] - 1536);
        reproj_screws[1] = 0.194 * (reproj_screws[1] - 1536);
        reproj_screws[2] = 0;
        std::pair<std::string, Pt3> ld2_3D(n.first, reproj_screws);
        reproj_screws_fcsv.insert(ld2_3D);
      }

      WriteFCSVFileFromNamePtMap(output_path + "/reproj_screws" + exp_ID + ".fcsv", reproj_screws_fcsv);
    }

    if( reproj_drr )
    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_model(default_cam);
      ray_caster->use_proj_store_replace_method();
      ray_caster->set_volume(device_vol);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();
      ray_caster->xform_cam_to_itk_phys(0) = init_cam_to_device;
      ray_caster->compute(0);

      WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/device_reproj" + exp_ID + ".png");
    }
  }

  return 0;
}
