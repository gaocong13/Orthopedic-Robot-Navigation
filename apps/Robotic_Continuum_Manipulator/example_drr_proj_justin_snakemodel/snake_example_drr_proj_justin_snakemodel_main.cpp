
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
constexpr bool kSAVE_SNAKE_VOL_TO_DISK = false;

using size_type = std::size_t;
using CamModelList = std::vector<CameraModel>;

const std::string snakemodel_huatt_path = "/home/cong/Research/Snake_Registration/Simulation_JustinNewSnake/JustinSnakeModel_huatt";

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Create an example DRR projection for Justin's new snake model");
  po.set_arg_usage("< snake model path > < output path >");
  po.set_min_num_pos_args(2);

  po.add("build-snake-model", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "build-snake-model", "Build snake model from segments and save built model to disk.")
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

  typedef itk::Image<float, 3> SnakeVolumeType;

  const std::string snake_model_path   = po.pos_args()[0];  // 3D spine landmarks path
  const std::string output_path        = po.pos_args()[1];  // Output path
  const size_type num_snake_vols       = 28;

  const bool kBUILD_SNAKE_MODEL        = po.get("build-snake-model");

  if( kBUILD_SNAKE_MODEL )
  {
    vout << "[Main] - Start building snake model from segments..." << std::endl;
    std::unique_ptr<BuildSnakeModel> buildsnake(new BuildSnakeModel(snake_model_path, 0, 0, 3.0, vout));

    // BuildSnakeModel buildsnake(snake_model_path, 0, 0, 0, vout);

    auto built_model = buildsnake->run();
    const std::string built_model_path = output_path + "/built_model.nii.gz";

    vout << "[Main] - Writing built snake model to disk..." << std::endl;
    WriteITKImageToDisk(built_model.GetPointer(), built_model_path);
  }

  vout << "[Main] - Reading snake model sgements from disk...\n";
  std::vector<SnakeVolumeType::Pointer> snake_att_list;
  for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx)
  {
    vout << "   Loading vol: " << vol_idx << std::endl;
    if(kSAVE_SNAKE_VOL_TO_DISK)
    {
      auto snake_vol = ReadITKImageFromDisk<SnakeVolumeType>(snake_model_path + "/" + fmt::format("{:03d}.nrrd", vol_idx));

      VolPtr snake_att;
      {
        itk::ImageRegionIterator<Vol> it_metal(snake_vol, snake_vol->GetRequestedRegion());
        it_metal.GoToBegin();
        while( !it_metal.IsAtEnd()){
          if(it_metal.Get() == 1 ){
            it_metal.Set(8000);
          }
          else if( it_metal.Get() == 0.5 ){
            it_metal.Set(4000);
          }
          else if(it_metal.Get() == 0){
            it_metal.Set(-1000);
          }
          ++it_metal;
        }

        vout << "      End HU --> Att. ..." << std::endl;
        snake_att = HUToLinAtt(snake_vol.GetPointer());
      }

      snake_att_list.push_back(snake_att);

      WriteITKImageToDisk(snake_vol.GetPointer(), snakemodel_huatt_path + "/" + fmt::format("{:03d}_hu.nii.gz", vol_idx));
      WriteITKImageToDisk(snake_att.GetPointer(), snakemodel_huatt_path + "/" + fmt::format("{:03d}_att.nii.gz", vol_idx));
    }
    else
    {
      auto snake_att = ReadITKImageFromDisk<SnakeVolumeType>(snakemodel_huatt_path + "/" + fmt::format("{:03d}_att.nii.gz", vol_idx));
      snake_att_list.push_back(snake_att);
    }
  }

  vout << "[Main] - Loading snake notch rotation centers from FCSV file..." << std::endl;
  const std::string notch_rot_cen_fcsv_path = snake_model_path + "/notch_rot_cen_Justin.fcsv";
  auto notch_rot_cen_fcsv = ReadFCSVFileNamePtMap(notch_rot_cen_fcsv_path);
  ConvertRASToLPS(&notch_rot_cen_fcsv);

  std::vector<Pt3> notch_rot_cen_list;
  for(size_type vol_idx = 1; vol_idx < num_snake_vols; ++vol_idx)
  {
    auto vol_idx_name = fmt::format("{:03d}", vol_idx);
    vout << "   Loading rot center: " << vol_idx_name << std::endl;

    auto fcsv_finder = notch_rot_cen_fcsv.find(vol_idx_name);
    if(fcsv_finder != notch_rot_cen_fcsv.end())
    {
      Pt3 notch_rot_cen_pt = fcsv_finder->second;
      notch_rot_cen_list.push_back(notch_rot_cen_pt);
    }
    else
    {
      std::cerr << "ERROR: NOT FOUND notch rotation center: " << vol_idx << std::endl;
    }
  }

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);
  FrameTransform test_xform = FrameTransform::Identity();
  test_xform.matrix().block(0,0,3,3) = EulerRotY(90 * kDEG2RAD) * EulerRotZ(90 * kDEG2RAD);
  test_xform(0, 3) = -150;
  test_xform(1, 3) = 50;
  test_xform(2, 3) = 0;

  vout << "[Main] Writing test xform to disk..." << std::endl;
  WriteITKAffineTransform(output_path + "/test_xform.h5", test_xform);

  ProjDataF32 proj_data;

  vout << "[Main] Performing DRR projection using snake segments...\n";
  {
    auto ray_caster = LineIntRayCasterFromProgOpts(po);
    ray_caster->set_camera_model(default_cam);
    ray_caster->use_proj_store_replace_method();
    ray_caster->set_volumes(snake_att_list);
    // ray_caster->set_ray_step_size(0.5);
    ray_caster->set_num_projs(1);
    ray_caster->allocate_resources();

    Float accumX = 0.;
    Float accumY = 0.;
    Float transX = 0.;
    Float transY = 0.;
    Float rotZ = 0.;
    Float fixed_ang = 5.0;
    for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx)
    {
      // ray_caster->use_proj_store_replace_method();
      FrameTransform cur_vol_xform = test_xform;
      if(vol_idx > 0)
      {
        vout << "      Transforming vol: " << vol_idx << std::endl;
        FrameTransform cur_ref_xform = FrameTransform::Identity();
        cur_ref_xform(0, 3) = -notch_rot_cen_list[vol_idx-1][0];
        cur_ref_xform(1, 3) = -notch_rot_cen_list[vol_idx-1][1];
        cur_ref_xform(2, 3) = -notch_rot_cen_list[vol_idx-1][2];

        rotZ += fixed_ang;
        Float dist_rot_cen = vol_idx > 1 ? (notch_rot_cen_list[vol_idx-1] - notch_rot_cen_list[vol_idx-2]).norm() : 0.;//TODO: rot center list index
        accumX += dist_rot_cen * sin(rotZ * kDEG2RAD);
        accumY += dist_rot_cen * cos(rotZ * kDEG2RAD);
        Float rotcenX = notch_rot_cen_list[vol_idx-1][0] - notch_rot_cen_list[0][0];
        Float rotcenY = notch_rot_cen_list[vol_idx-1][1] - notch_rot_cen_list[0][1];
        Float accumX_wrt_rotcen = accumX - rotcenX;
        Float accumY_wrt_rotcen = accumY - rotcenY;
        transX = accumY_wrt_rotcen * sin(rotZ * kDEG2RAD) - accumX_wrt_rotcen * cos(rotZ * kDEG2RAD);
        transY = -(accumX_wrt_rotcen * sin(rotZ * kDEG2RAD) + accumY_wrt_rotcen * cos(rotZ * kDEG2RAD));
        transX = vol_idx % 2 == 0 ? transX + 0.03 : transX - 0.03;

        cur_vol_xform = cur_ref_xform.inverse() * EulerRotXYZTransXYZFrame(0, 0, rotZ * kDEG2RAD, transX, transY, 0) * cur_ref_xform * cur_vol_xform;
      }
      vout << "   Projecting vol: " << vol_idx << std::endl;
      ray_caster->distribute_xform_among_cam_models(cur_vol_xform);
      ray_caster->compute(vol_idx);
      ray_caster->use_proj_store_accum_method();
    }

    vout << "[Main] Writing projected DRR image to disk ...\n";
    proj_data.img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 5000);
    proj_data.cam = default_cam;

    WriteITKImageRemap8bpp(proj_data.img.GetPointer(), output_path + "/snake_drr_proj.png");
  }

  vout << "[Main] Writing proj data to disk ...\n";
  const std::string proj_data_h5_path = output_path + "/proj_data.h5";
  H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
  WriteProjDataH5(proj_data, &h5);

  vout << "exiting...\n";

  return 0;
}
