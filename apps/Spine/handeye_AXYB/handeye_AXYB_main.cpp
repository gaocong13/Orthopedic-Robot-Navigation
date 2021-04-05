
// STD
#include <iostream>
#include <vector>

#include <fmt/format.h>

#include "xregProgOptUtils.h"
#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregLandmarkMapUtils.h"
#include "xregAnatCoordFrames.h"
#include "xregH5ProjDataIO.h"
#include "xregPnPUtils.h"
#include "xregHDF5.h"
#include "xregCIOSFusionDICOM.h"

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
  const std::string UR_kins_path              = po.pos_args()[1];
  const std::string pnp_xform_path            = po.pos_args()[2];
  const std::string exp_list_path             = po.pos_args()[3];  // Experiment list file path
  const std::string output_path               = po.pos_args()[4];  // Output path

  const std::string device_3d_fcsv_path       = meta_data_path + "/Device3Dlandmark.fcsv";

  std::cout << "reading device 3D landmarks from FCSV file..." << std::endl;
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

  std::vector <vctFrm4x4> A_frames;    //< Transformation of A frames
  std::vector <vctFrm4x4> B_frames;    //< Transformation of B frames

  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string exp_ID                  = exp_ID_list[idx];

    // Read Robot End Effector transformation from h5_slicer file
    FrameTransform UReef_xform;
    {
      const std::string src_ureef_path          = UR_kins_path + "/" + exp_ID + "/ur_eef.h5";
      H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
      H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
      H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
      std::vector<float> UReef_tracker          = ReadVectorH5Float("TranformParameters", ureef_group0);
      UReef_xform                               = ConvertSlicerToITK(UReef_tracker);
    }

    const std::string src_pnp_path            = pnp_xform_path + "/device_pnp_xform" + exp_ID + ".h5";
    FrameTransform pnp_xform =  ReadITKAffineTransformFromFile(src_pnp_path);

    FrameTransform device_cam_to_ref = device_rotcen_ref * pnp_xform;
    FrameTransform device_ref_to_cam = device_cam_to_ref.inverse(); //actuation jig relative to C-arm

    vctFrm4x4 A_frame;
    vctFrm4x4 B_frame;

    for(size_type idx=0; idx<4; ++idx)
    {
      for(size_type idy=0; idy<4; ++idy)
      {
        A_frame[idx][idy] = UReef_xform(idx, idy);
        B_frame[idx][idy] = device_ref_to_cam(idx, idy);
      }
    }


    std::cout << exp_ID << std::endl;
    std::cout << "A frame:\n" << A_frame << std::endl;
    std::cout << "B frame:\n" << B_frame << std::endl;
    std::cout << " ***************************************** " << std::endl;

    A_frames.push_back(A_frame);
    B_frames.push_back(B_frame);
  }

  if(A_frames.size() <= 5){
    std::cerr << "At least 5 frames are required for hand-eye calibration" << std::endl;
    return kEXIT_VAL_BAD_USE;
  }

  vctDoubleMat AX, BX, AY, BY;
  AX.SetSize(2*A_frames.size()*(A_frames.size()+1), 4);
  BX.SetSize(2*B_frames.size()*(B_frames.size()+1), 4);
  AY.SetSize(2*A_frames.size()*(A_frames.size()+1), 4);
  BY.SetSize(2*B_frames.size()*(B_frames.size()+1), 4);

  // precalculate inverse to solve for X
  unsigned int count = 0;
  for (unsigned int i=0; i<A_frames.size(); i++)
  {
    for (unsigned int j=i; j<A_frames.size(); j++)
    {
      AX.Ref(4, 4, 4*count, 0).Assign ( vctDoubleMat (A_frames[j].Inverse() * A_frames[i]));
      BX.Ref(4, 4, 4*count, 0).Assign ( vctDoubleMat (B_frames[j].Inverse() * B_frames[i]));
      AY.Ref(4, 4, 4*count, 0).Assign ( vctDoubleMat (A_frames[j] * A_frames[i].Inverse()));
      BY.Ref(4, 4, 4*count, 0).Assign ( vctDoubleMat (B_frames[j] * B_frames[i].Inverse()));
      ++count;
    }
  }

  vctFrm4x4 X, Y;
  BIGSS::ax_xb(AX, BX, X);
  BIGSS::ax_xb(AY, BY, Y);

  // print out results
  std::cout << "X = " << std::endl << X << std::endl;
  std::cout << "Y = " << std::endl << Y << std::endl;

  // Save hand-eye result to file
  FrameTransform pnphandeye_X = FrameTransform::Identity();
  FrameTransform pnphandeye_Y = FrameTransform::Identity();
  for(size_type idx=0; idx<4; ++idx)
  {
    for(size_type idy=0; idy<4; ++idy)
    {
      pnphandeye_X(idx, idy) = X[idx][idy];
      pnphandeye_Y(idx, idy) = Y[idx][idy];
    }
  }

  const std::string pnphandeye_X_file = output_path + "/pnp_handeye_X.h5";
  const std::string pnphandeye_Y_file = output_path + "/pnp_handeye_Y.h5";
  WriteITKAffineTransform(pnphandeye_X_file, pnphandeye_X);
  WriteITKAffineTransform(pnphandeye_Y_file, pnphandeye_Y);

  return 0;
}
