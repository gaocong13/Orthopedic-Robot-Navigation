/*
 * MIT License
 *
 * Copyright (c) 2020 Robert Grupp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <fmt/format.h>

#include "xregProgOptUtils.h"
#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregLandmarkMapUtils.h"
#include "xregAnatCoordFrames.h"
#include "xregCIOSFusionDICOM.h"
#include "xregSampleUtils.h"
#include "xregSampleUniformUnitVecs.h"
#include "xregRotUtils.h"
#include "xregRigidUtils.h"
#include "xregH5ProjDataIO.h"
#include "xregOpenCVUtils.h"
#include "xregHDF5.h"
#include "xregProjPreProc.h"
#include "xregRayCastProgOpts.h"
#include "xregRayCastInterface.h"
#include "xregHUToLinAtt.h"
#include "xregImageAddPoissonNoise.h"

using namespace xreg;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;

int main(int argc, char* argv[])
{
  // First, set up the program options

  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Generate H5 projection data of the calibration images");

  po.set_arg_usage("<exp ID list> <pnp path> <img path> <output path>");
  po.set_min_num_pos_args(4);

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

  const std::string exp_list_path   = po.pos_args()[0];
  const std::string pnp_path        = po.pos_args()[1];
  const std::string img_path        = po.pos_args()[2];
  const std::string output_path     = po.pos_args()[3];

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

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

  const size_type num_imgs = lineNumber;

  ProjDataF32List pjs;
  pjs.resize(num_imgs);

  FrameTransformList pnp_xform_list;

  for(size_type img_idx = 0; img_idx < num_imgs; ++img_idx)
  {
    const std::string img_ID = exp_ID_list[img_idx];

    FrameTransformList pel_xform_list;

    std::cout << "running img: " << img_ID << std::endl;

    const std::string src_pnp_path = pnp_path + "/pnp_xform" + img_ID + ".h5";
    auto pnp_xform                 = ReadITKAffineTransformFromFile(src_pnp_path);

    pnp_xform_list.push_back(pnp_xform);

    const std::string src_img_path = img_path + "/" + img_ID;
    std::vector<CIOSFusionDICOMInfo> spinecios_metas(1);
    std::tie(pjs[img_idx].img, spinecios_metas[0]) = ReadCIOSFusionDICOMFloat(src_img_path);
  }

  std::vector<CameraModel> orig_cams;
  for (auto& pd : pjs)
  {
    orig_cams.push_back(default_cam);
  }

  // Using device regi as fiducial
  auto cams_devicefid = CreateCameraWorldUsingFiducial(orig_cams, pnp_xform_list);

  for(size_type cam_idx = 0; cam_idx < num_imgs; ++cam_idx)
  {
    pjs[cam_idx].cam = cams_devicefid[cam_idx];
  }

  const std::string proj_data_h5_path = output_path + "/projs.h5";
  H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);

  WriteProjDataH5(pjs, &h5);

  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
