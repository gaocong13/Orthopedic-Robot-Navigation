
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
#include "xregHUToLinAtt.h"
#include "xregProjPreProc.h"
#include "xregCIOSFusionDICOM.h"
#include "xregHDF5.h"
#include "xregHUToLinAtt.h"
#include "xregImageAddPoissonNoise.h"
#include "xregRotUtils.h"
#include "xregRigidUtils.h"

#include "xregRigidUtils.h"

#include "bigssMath.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVersor.h"
#include "itkChangeInformationImageFilter.h"

#include "xregBuildJustinSnakeModel.h"

using namespace xreg;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;
constexpr bool kSAVE_REGI_DEBUG = false;

using size_type = std::size_t;
using CamModelList = std::vector<CameraModel>;

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Create an example DRR projection for Justin's new snake model");
  po.set_arg_usage("< snake model path > < output path >");
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

  typedef itk::Image<float, 3> SnakeVolumeType;

  const std::string snake_model_path   = po.pos_args()[0];  // 3D spine landmarks path
  const std::string output_path        = po.pos_args()[1];  // Output path
  const size_type num_snake_vols       = 28;

  std::unique_ptr<BuildSnakeModel> buildsnake(new BuildSnakeModel(snake_model_path, 0, 0, 0, vout));

  // BuildSnakeModel buildsnake(snake_model_path, 0, 0, 0, vout);

  auto built_model = buildsnake->run();
  const std::string built_model_path = output_path + "/built_model.nii.gz";
  WriteITKImageToDisk(built_model.GetPointer(), built_model_path);

  return 0;


  vout << "Reading snake models...\n";
  std::vector<SnakeVolumeType::Pointer> snake_att_list;
  for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx)
  {
    auto snake_vol = ReadITKImageFromDisk<SnakeVolumeType>(snake_model_path + "/" + fmt::format("{:03d} cropped.nrrd", vol_idx));
/*
    // HU value manipulation
    itk::ImageRegionIterator<SnakeVolumeType> it_metal(snake_vol, snake_vol->GetRequestedRegion());
    it_metal.GoToBegin();
    while( !it_metal.IsAtEnd()){
      if(it_metal.Get() == 1){
        it_metal.Set(8000);
      }
      else if(it_metal.Get() == 0){
        it_metal.Set(-1000);
      }
      ++it_metal;
    }

    auto hu2att_snakemetal = HUToLinAtt(snake_vol.GetPointer());

    snake_att_list.push_back(hu2att_snakemetal);
*/
    snake_att_list.push_back(snake_vol);
  }

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);
  FrameTransform test_xform = FrameTransform::Identity();
  test_xform.matrix().block(0,0,3,3) = EulerRotY(90 * kDEG2RAD) * EulerRotZ(90 * kDEG2RAD);
  test_xform(0, 3) = -150;
  test_xform(1, 3) = 50;
  test_xform(2, 3) = 300;

  WriteITKAffineTransform(output_path + "/test_xform.h5", test_xform);

  ProjDataF32 proj_data;

  vout << "Doing projection ...\n";
  {
    auto ray_caster = LineIntRayCasterFromProgOpts(po);
    ray_caster->set_camera_model(default_cam);
    ray_caster->use_proj_store_replace_method();
    ray_caster->set_volumes(snake_att_list);
    // ray_caster->set_ray_step_size(0.5);
    ray_caster->set_num_projs(1);
    ray_caster->allocate_resources();

    for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx)
    {
      vout << "   Projecting vol: " << vol_idx << std::endl;
      // ray_caster->use_proj_store_replace_method();
      ray_caster->distribute_xform_among_cam_models(test_xform);
      ray_caster->compute(vol_idx);
      ray_caster->use_proj_store_accum_method();
    }

    vout << "Writing img to disk ...\n";
    WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/snake_drr_proj.png");
    proj_data.img = ray_caster->proj(0).GetPointer();
    proj_data.cam = default_cam;
  }

  const std::string proj_data_h5_path = output_path + "/proj_data.h5";
  H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
  WriteProjDataH5(proj_data, &h5);

  vout << "exiting...\n";

  return 0;
}
