
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

  po.set_help("Calculate Forward Kinematics using Injector Body marker and Base Marker");
  po.set_arg_usage("<Slicer path> <Image ID list txt file path> <Output folder path>");
  po.set_min_num_pos_args(3);

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

  const std::string Slicer_path               = po.pos_args()[0];
  const std::string exp_list_path             = po.pos_args()[1];  // Experiment list file path
  const std::string output_path               = po.pos_args()[2];  // Output path

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

  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string exp_ID                  = exp_ID_list[idx];

    FrameTransform Injector_xform;
    {
      const std::string src_Injector_path          = Slicer_path + "/" + exp_ID + "/InjectorBody.h5";
      H5::H5File h5_Injector(src_Injector_path, H5F_ACC_RDWR);
      H5::Group Injector_transform_group           = h5_Injector.openGroup("TransformGroup");
      H5::Group Injector_group0                    = Injector_transform_group.openGroup("0");
      std::vector<float> Injector_tracker          = ReadVectorH5Float("TranformParameters", Injector_group0);
      Injector_xform                               = ConvertSlicerToITK(Injector_tracker);
    }

    FrameTransform StyroBase_xform;
    {
      const std::string src_StyroBase_path          = Slicer_path + "/" + exp_ID + "/StyroBase.h5";
      H5::H5File h5_StyroBase(src_StyroBase_path, H5F_ACC_RDWR);
      H5::Group StyroBase_transform_group           = h5_StyroBase.openGroup("TransformGroup");
      H5::Group StyroBase_group0                    = StyroBase_transform_group.openGroup("0");
      std::vector<float> StyroBase_tracker          = ReadVectorH5Float("TranformParameters", StyroBase_group0);
      StyroBase_xform                               = ConvertSlicerToITK(StyroBase_tracker);
    }

    FrameTransform FK_xform = StyroBase_xform.inverse() * Injector_xform;

    const std::string dir_name = output_path + "/" + exp_ID;

    const std::string FK_filename = dir_name + "/ur_eef.h5";
    WriteITKAffineTransform(FK_filename, FK_xform);
  }

  return 0;
}
