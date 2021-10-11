
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
#include "xregHUToLinAtt.h"
#include "xregProjPreProc.h"
#include "xregPnPUtils.h"
#include "xregFoldNormDist.h"
#include "xregHDF5.h"
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

struct RegiErrorInfo
{
    CoordScalar rot_error_deg;
    CoordScalar trans_error;

    CoordScalar rot_x_error_deg;
    CoordScalar rot_y_error_deg;
    CoordScalar rot_z_error_deg;

    CoordScalar trans_x_error;
    CoordScalar trans_y_error;
    CoordScalar trans_z_error;
};

void ExportRegiErrorToDisk(FrameTransform regi_err_xform, const std::string file_name, const std::string exp_ID)
{
  RegiErrorInfo err;
  std::tie(err.rot_x_error_deg, err.rot_y_error_deg, err.rot_z_error_deg, err.trans_x_error, err.trans_y_error, err.trans_z_error) = RigidXformToEulerXYZAndTrans(regi_err_xform);
  err.rot_x_error_deg *= kRAD2DEG; err.rot_y_error_deg *= kRAD2DEG; err.rot_z_error_deg *= kRAD2DEG;

  std::ofstream regi_err;
  regi_err.open(file_name, std::ios::app);
  regi_err << "ExpID"  << ',' << exp_ID
           << ',' << err.trans_x_error       << ',' << err.trans_y_error       << ',' << err.trans_z_error
           << ',' << err.rot_x_error_deg     << ',' << err.rot_y_error_deg     << ',' << err.rot_z_error_deg    << '\n';
  regi_err.close();
}

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Simulation of Single-view Spine Registration");
  po.set_arg_usage("< meta data path > < source h5 path > < output path >");
  po.set_min_num_pos_args(2);

  po.add("is-multiview", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "is-multiview",
         "Registration is multi-view, default is False")
    << false;

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

  const std::string meta_data_path   = po.pos_args()[0];  // 3D spine landmarks path
  const std::string src_h5_path      = po.pos_args()[1];
  const std::string output_path      = po.pos_args()[2];  // Output path

  const std::string spinevol_path = meta_data_path + "/Spine21-2512_CT_crop.nrrd";
  const std::string spineseg_path = meta_data_path + "/Spine21-2512_seg_crop.nrrd";
  const std::string sacrumseg_path = meta_data_path + "/Spine21-2512_sacrum_seg_crop.nrrd";
  const std::string spine_gt_xform_path = meta_data_path + "/spine_gt_xform.h5";
  const std::string device_gt_xform_path = meta_data_path + "/device_gt_xform.h5";
  const std::string spine_3d_fcsv_path = meta_data_path + "/Spine_3D_landmarks.fcsv";

  const std::string device_3d_fcsv_path    = meta_data_path + "/Device3Dlandmark.fcsv";
  const std::string device_3d_bb_fcsv_path = meta_data_path + "/Device3Dbb.fcsv";
  const std::string devicevol_path         = meta_data_path + "/Device_crop_CT.nii.gz";
  const std::string deviceseg_path         = meta_data_path + "/Device_crop_seg.nii.gz";

  const bool is_multi_view = po.get("is-multiview");

  vout << "reading spine anatomical landmarks from FCSV file..." << std::endl;
  auto spine_3d_fcsv = ReadFCSVFileNamePtMap(spine_3d_fcsv_path);
  ConvertRASToLPS(&spine_3d_fcsv);

  vout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_fcsv = ReadFCSVFileNamePtMap(device_3d_fcsv_path);
  ConvertRASToLPS(&device_3d_fcsv);

  FrameTransform vert1_ref_frame = FrameTransform::Identity();
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

    vert1_ref_frame(0, 3) = -rotcenter[0];
    vert1_ref_frame(1, 3) = -rotcenter[1];
    vert1_ref_frame(2, 3) = -rotcenter[2];
  }

  FrameTransform vert2_ref_frame = FrameTransform::Identity();
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

    vert2_ref_frame(0, 3) = -rotcenter[0];
    vert2_ref_frame(1, 3) = -rotcenter[1];
    vert2_ref_frame(2, 3) = -rotcenter[2];
  }

  FrameTransform vert3_ref_frame = FrameTransform::Identity();
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

    vert3_ref_frame(0, 3) = -rotcenter[0];
    vert3_ref_frame(1, 3) = -rotcenter[1];
    vert3_ref_frame(2, 3) = -rotcenter[2];
  }

  FrameTransform vert4_ref_frame = FrameTransform::Identity();
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

    vert4_ref_frame(0, 3) = -rotcenter[0];
    vert4_ref_frame(1, 3) = -rotcenter[1];
    vert4_ref_frame(2, 3) = -rotcenter[2];
  }

  FrameTransform sacrum_ref_frame = FrameTransform::Identity();
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

    sacrum_ref_frame(0, 3) = -rotcenter[0];
    sacrum_ref_frame(1, 3) = -rotcenter[1];
    sacrum_ref_frame(2, 3) = -rotcenter[2];
  }

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

    spine_ref_frame(0, 3) = -spine_rotcenter[0];
    spine_ref_frame(1, 3) = -spine_rotcenter[1];
    spine_ref_frame(2, 3) = -spine_rotcenter[2];
  }

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

    device_ref_frame(0, 3) = -device_rotcenter[0];
    device_ref_frame(1, 3) = -device_rotcenter[1];
    device_ref_frame(2, 3) = -device_rotcenter[2];
  }

  for(int idx = 1; idx < 1000; ++idx)
  {
    const std::string exp_ID = fmt::format("{:04d}", idx);
    vout << "opening source H5 for reading: " << exp_ID << std::endl;
    H5::H5File h5(src_h5_path + "/" + exp_ID + ".h5", H5F_ACC_RDONLY);
    const FrameTransform gt_cam_wrt_spine = ReadAffineTransform4x4H5("gt-cam-wrt-spine", h5);
    const FrameTransform gt_cam_wrt_device = ReadAffineTransform4x4H5("gt-cam-wrt-device", h5);
    const FrameTransform regi_cam_wrt_rigid_spine = ReadAffineTransform4x4H5("regi-cam-wrt-rigid-spine", h5);
    const FrameTransform regi_cam_wrt_device = ReadAffineTransform4x4H5("regi-cam-wrt-device", h5);
    const FrameTransform regi_cam_wrt_sacrum = ReadAffineTransform4x4H5("regi-cam-wrt-sacrum", h5);
    const FrameTransform regi_cam_wrt_vert1 = ReadAffineTransform4x4H5("regi-cam-wrt-vert1", h5);
    const FrameTransform regi_cam_wrt_vert2 = ReadAffineTransform4x4H5("regi-cam-wrt-vert2", h5);
    const FrameTransform regi_cam_wrt_vert3 = ReadAffineTransform4x4H5("regi-cam-wrt-vert3", h5);
    const FrameTransform regi_cam_wrt_vert4 = ReadAffineTransform4x4H5("regi-cam-wrt-vert4", h5);
    if(is_multi_view)
    {
      const FrameTransform fid_cam_wrt_device = ReadAffineTransform4x4H5("fid-cam-wrt-device", h5);

      const FrameTransform err_rigid_spine_ref_wrt_cam = spine_ref_frame * regi_cam_wrt_rigid_spine * fid_cam_wrt_device * gt_cam_wrt_spine.inverse() * spine_ref_frame.inverse();
      const FrameTransform err_device_ref_wrt_cam = device_ref_frame * regi_cam_wrt_device * fid_cam_wrt_device * gt_cam_wrt_device.inverse() * device_ref_frame.inverse();
      const FrameTransform err_vert1_ref_wrt_cam = vert1_ref_frame * regi_cam_wrt_vert1 * fid_cam_wrt_device * gt_cam_wrt_spine.inverse() * vert1_ref_frame.inverse();
      const FrameTransform err_vert2_ref_wrt_cam = vert2_ref_frame * regi_cam_wrt_vert2 * fid_cam_wrt_device * gt_cam_wrt_spine.inverse() * vert2_ref_frame.inverse();
      const FrameTransform err_vert3_ref_wrt_cam = vert3_ref_frame * regi_cam_wrt_vert3 * fid_cam_wrt_device * gt_cam_wrt_spine.inverse() * vert3_ref_frame.inverse();
      const FrameTransform err_vert4_ref_wrt_cam = vert4_ref_frame * regi_cam_wrt_vert4 * fid_cam_wrt_device * gt_cam_wrt_spine.inverse() * vert4_ref_frame.inverse();
      const FrameTransform err_sacrum_ref_wrt_cam = sacrum_ref_frame * regi_cam_wrt_sacrum * fid_cam_wrt_device * gt_cam_wrt_spine.inverse() * sacrum_ref_frame.inverse();

      ExportRegiErrorToDisk(err_rigid_spine_ref_wrt_cam, output_path + "/regi_error_rigid_spine.txt", exp_ID);
      ExportRegiErrorToDisk(err_device_ref_wrt_cam, output_path + "/regi_error_device.txt", exp_ID);
      ExportRegiErrorToDisk(err_vert1_ref_wrt_cam, output_path + "/regi_error_vert1.txt", exp_ID);
      ExportRegiErrorToDisk(err_vert2_ref_wrt_cam, output_path + "/regi_error_vert2.txt", exp_ID);
      ExportRegiErrorToDisk(err_vert3_ref_wrt_cam, output_path + "/regi_error_vert3.txt", exp_ID);
      ExportRegiErrorToDisk(err_vert4_ref_wrt_cam, output_path + "/regi_error_vert4.txt", exp_ID);
      ExportRegiErrorToDisk(err_sacrum_ref_wrt_cam, output_path + "/regi_error_sacrum.txt", exp_ID);
    }
    else
    {
      const FrameTransform err_rigid_spine_ref_wrt_cam = spine_ref_frame * regi_cam_wrt_rigid_spine * gt_cam_wrt_spine.inverse() * spine_ref_frame.inverse();
      const FrameTransform err_device_ref_wrt_cam = device_ref_frame * regi_cam_wrt_device * gt_cam_wrt_device.inverse() * device_ref_frame.inverse();
      const FrameTransform err_vert1_ref_wrt_cam = vert1_ref_frame * regi_cam_wrt_vert1 * gt_cam_wrt_spine.inverse() * vert1_ref_frame.inverse();
      const FrameTransform err_vert2_ref_wrt_cam = vert2_ref_frame * regi_cam_wrt_vert2 * gt_cam_wrt_spine.inverse() * vert2_ref_frame.inverse();
      const FrameTransform err_vert3_ref_wrt_cam = vert3_ref_frame * regi_cam_wrt_vert3 * gt_cam_wrt_spine.inverse() * vert3_ref_frame.inverse();
      const FrameTransform err_vert4_ref_wrt_cam = vert4_ref_frame * regi_cam_wrt_vert4 * gt_cam_wrt_spine.inverse() * vert4_ref_frame.inverse();
      const FrameTransform err_sacrum_ref_wrt_cam = sacrum_ref_frame * regi_cam_wrt_sacrum * gt_cam_wrt_spine.inverse() * sacrum_ref_frame.inverse();

      ExportRegiErrorToDisk(err_rigid_spine_ref_wrt_cam, output_path + "/regi_error_rigid_spine.txt", exp_ID);
      ExportRegiErrorToDisk(err_device_ref_wrt_cam, output_path + "/regi_error_device.txt", exp_ID);
      ExportRegiErrorToDisk(err_vert1_ref_wrt_cam, output_path + "/regi_error_vert1.txt", exp_ID);
      ExportRegiErrorToDisk(err_vert2_ref_wrt_cam, output_path + "/regi_error_vert2.txt", exp_ID);
      ExportRegiErrorToDisk(err_vert3_ref_wrt_cam, output_path + "/regi_error_vert3.txt", exp_ID);
      ExportRegiErrorToDisk(err_vert4_ref_wrt_cam, output_path + "/regi_error_vert4.txt", exp_ID);
      ExportRegiErrorToDisk(err_sacrum_ref_wrt_cam, output_path + "/regi_error_sacrum.txt", exp_ID);
    }

    h5.close();
  }
  return 0;
}
