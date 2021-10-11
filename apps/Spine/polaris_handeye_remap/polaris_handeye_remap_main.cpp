
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

  const std::string refpolaris_xform_file_path = po.pos_args()[0];
  const std::string refUR_kins_path            = po.pos_args()[1];
  const std::string handeye_regi_file_path     = po.pos_args()[2];
  const std::string UR_kins_path               = po.pos_args()[3];
  const std::string exp_list_path              = po.pos_args()[4];  // Experiment list file path
  const std::string output_path                = po.pos_args()[5];  // Output path
  const std::string Jigname                    = po.pos_args()[6];  // Polaris Jig Name

  const bool cal_3D_pts = true; //Calculate 3D URee and Device Tip points in URbase and write to txt file.

  FrameTransform refUReef_xform;
  {
    const std::string src_ureef_path          = refUR_kins_path;
    H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
    H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
    H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
    std::vector<float> UReef_tracker          = ReadVectorH5Float("TranformParameters", ureef_group0);
    refUReef_xform                            = ConvertSlicerToITK(UReef_tracker);
  }

  FrameTransform ref_polaris_xform;
  {
    H5::H5File h5_tracker(refpolaris_xform_file_path, H5F_ACC_RDWR);
    H5::Group tracker_transform_group           = h5_tracker.openGroup("TransformGroup");
    H5::Group tracker_group0                    = tracker_transform_group.openGroup("0");
    std::vector<float> tracker_vec              = ReadVectorH5Float("TranformParameters", tracker_group0);
    ref_polaris_xform                           = ConvertSlicerToITK(tracker_vec);
  }

  FrameTransform handeye_regi_X = ReadITKAffineTransformFromFile(handeye_regi_file_path);

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


  Pt3 polaris_origin = {0, 0, 0};
  Pt3 UR_origin = {0, 0, 0};
  Pt3 UR_ref_origin = refUReef_xform * UR_origin;

  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string exp_ID                = exp_ID_list[idx];

    std::cout << "Running..." << exp_ID << std::endl;

    FrameTransform UReef_xform;
    {
      const std::string src_ureef_path          = UR_kins_path + "/" + exp_ID_list[idx] + "/ur_eef.h5";
      H5::H5File h5_ureef(src_ureef_path, H5F_ACC_RDWR);
      H5::Group ureef_transform_group           = h5_ureef.openGroup("TransformGroup");
      H5::Group ureef_group0                    = ureef_transform_group.openGroup("0");
      std::vector<float> UReef_tracker          = ReadVectorH5Float("TranformParameters", ureef_group0);
      UReef_xform                               = ConvertSlicerToITK(UReef_tracker);
    }

    FrameTransform cal_tracker_to_polaris_xform = handeye_regi_X.inverse() * UReef_xform.inverse() * refUReef_xform * handeye_regi_X * ref_polaris_xform.inverse();

    FrameTransform act_polaris_to_tracker_xform;
    {
      const std::string src_tracker_path          = UR_kins_path + "/" + exp_ID_list[idx] + "/" + Jigname + ".h5";
      H5::H5File h5_tracker(src_tracker_path, H5F_ACC_RDWR);
      H5::Group tracker_transform_group           = h5_tracker.openGroup("TransformGroup");
      H5::Group tracker_group0                    = tracker_transform_group.openGroup("0");
      std::vector<float> tracker_vec              = ReadVectorH5Float("TranformParameters", tracker_group0);
      act_polaris_to_tracker_xform                = ConvertSlicerToITK(tracker_vec);
    }

    Pt3 UR_cal_origin = UReef_xform * UR_origin;

    Pt3 UR_origin_diff = UR_cal_origin - UR_ref_origin;

    if (cal_3D_pts)
    {
      Pt3 act_polaris_origin_wrt_tracker = act_polaris_to_tracker_xform * polaris_origin;
      Pt3 cal_polaris_origin_wrt_tracker = cal_tracker_to_polaris_xform.inverse() * polaris_origin;

      std::ofstream Tracker_ref_3Dpt;
      Tracker_ref_3Dpt.open(output_path + "/Tracker_ref_3Dpt.txt", std::ios::app);
      Tracker_ref_3Dpt << exp_ID << "," << "ActPolarisOrigin," << act_polaris_origin_wrt_tracker[0] << "," << act_polaris_origin_wrt_tracker[1] << "," << act_polaris_origin_wrt_tracker[2] << ","
                      << "PolarisTip," << cal_polaris_origin_wrt_tracker[0] << "," << cal_polaris_origin_wrt_tracker[1] << "," << cal_polaris_origin_wrt_tracker[2] << ","
                      << "RelativeUROrigin," << UR_origin_diff.norm() << '\n';
      Tracker_ref_3Dpt.close();
    }
  }

  return 0;
}
