
// STD
#include <iostream>
#include <vector>

#include <fmt/format.h>
#include <itkFlipImageFilter.h>

#include "xregProgOptUtils.h"
#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregITKLabelUtils.h"
#include "xregAnatCoordFrames.h"
#include "xregH5ProjDataIO.h"
#include "xregRayCastProgOpts.h"
#include "xregRayCastInterface.h"
#include "xregHUToLinAtt.h"
#include "xregProjPreProc.h"
#include "xregCIOSFusionDICOM.h"
#include "xregHipSegUtils.h"

#include "xregPAODrawBones.h"
#include "xregRigidUtils.h"

#include "bigssMath.h"

using namespace xreg;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;

using size_type = std::size_t;

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Apply Femur Segmentation to CT volume and save to disk");
  po.set_arg_usage("<Femur CT volume path> <Femur segmentation path> <Exported volume after applying segmentation to CT>");
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

  const std::string vol_path             = po.pos_args()[0];  // Femur CT volume path
  const std::string seg_path             = po.pos_args()[1];  // Femur segmentation path
  const std::string output_path          = po.pos_args()[2];  // Exported volume after applying segmentation to CT

  vout << "reading CT ..." << std::endl;
  auto vol_hu = ReadITKImageFromDisk<RayCaster::Vol>(vol_path);
  using CTVolumeType = RayCaster::Vol;

  CTVolumeType::Pointer flip_vol_hu = vol_hu;
  {
    using ImageFlipper = itk::FlipImageFilter<CTVolumeType>;
    ImageFlipper::Pointer flipper = ImageFlipper::New();
    flipper->SetInput(vol_hu);
    ImageFlipper::FlipAxesArrayType axes_to_flip;
    axes_to_flip[0] = false;
    axes_to_flip[1] = false;
    axes_to_flip[2] = true;
    flipper->SetFlipAxes(axes_to_flip);
    flipper->Update();
    flip_vol_hu = flipper->GetOutput();
  }

  vout << "reading segmentation ..." << std::endl;
  using SegVolumeType = itk::Image<unsigned char,3>;
  auto vol_seg = ReadITKImageFromDisk<SegVolumeType>(seg_path);

  SegVolumeType::Pointer flip_vol_seg = vol_seg;
  {
    using ImageFlipper = itk::FlipImageFilter<SegVolumeType>;
    ImageFlipper::Pointer flipper = ImageFlipper::New();
    flipper->SetInput(vol_seg);
    ImageFlipper::FlipAxesArrayType axes_to_flip;
    axes_to_flip[0] = false;
    axes_to_flip[1] = false;
    axes_to_flip[2] = true;
    flipper->SetFlipAxes(axes_to_flip);
    flipper->Update();
    flip_vol_seg = flipper->GetOutput();
  }

  unsigned char femur_label = 1;
  vout << "applying segmentation to volume ..." << std::endl;
  auto femur_hu_vol = ApplyMaskToITKImage(flip_vol_hu.GetPointer(), flip_vol_seg.GetPointer(), femur_label, float(0), true);

  WriteITKImageToDisk(flip_vol_hu.GetPointer(),  output_path + "/flipped_CT.nii.gz");
  WriteITKImageToDisk(flip_vol_seg.GetPointer(), output_path + "/flipped_Seg.nii.gz");
  WriteITKImageToDisk(femur_hu_vol.GetPointer(), output_path + "/flipped_SegCT.nii.gz");

  return kEXIT_VAL_SUCCESS;
}
