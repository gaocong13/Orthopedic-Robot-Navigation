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
#include "xregAnatCoordFrames.h"
#include "xregCIOSFusionDICOM.h"
#include "xregRotUtils.h"
#include "xregRigidUtils.h"
#include "xregH5ProjDataIO.h"
#include "xregOpenCVUtils.h"
#include "xregHDF5.h"

int main(int argc, char* argv[])
{
  using namespace xreg;

  constexpr int kEXIT_VAL_SUCCESS = 0;
  constexpr int kEXIT_VAL_BAD_USE = 1;

  // First, set up the program options

  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Write CT vol data to h5 file.");

  po.set_arg_usage("<Dst H5 File> <Vol path> <Spec ID 1> <Spec ID 2> ... <Spec ID N>>");
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

  const std::string dst_h5_path             = po.pos_args()[0];
  const std::string vol_3d_path             = po.pos_args()[1];

  std::vector<std::string> spec_ID_list(po.pos_args().begin() + 2, po.pos_args().end());

  const size_type num_specs = spec_ID_list.size();

  vout << "opening output file for writing..." << std::endl;
  H5::H5File dst_h5(dst_h5_path, H5F_ACC_TRUNC);

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  // map world --> camera 3D
  const FrameTransform extrins = default_cam.extrins;

  // project camera 3D --> image plane 2D pixels
  const Mat3x3 intrins = default_cam.intrins;

  // Every projection should have the same extrinsics/intrinsics for this dataset
  vout << "writing camera params..." << std::endl;
  {
    H5::Group cam_g = dst_h5.createGroup("proj-params");

    WriteSingleScalarH5("num-rows", default_cam.num_det_rows, &cam_g);
    WriteSingleScalarH5("num-cols", default_cam.num_det_cols, &cam_g);

    WriteSingleScalarH5("pixel-row-spacing", default_cam.det_row_spacing, &cam_g);
    WriteSingleScalarH5("pixel-col-spacing", default_cam.det_col_spacing, &cam_g);

    WriteMatrixH5("intrinsic", intrins, &cam_g);
    WriteAffineTransform4x4("extrinsic", extrins, &cam_g);
  }

  for(size_type spec_idx = 0; spec_idx < num_specs; ++spec_idx)
  {
    const std::string spec_ID = spec_ID_list[spec_idx];
    vout << "reading input CT volume..." << std::endl;
    const std::string vol_3d_file = vol_3d_path + '/' + spec_ID + ".nrrd";
    auto vol_hu = ReadITKImageFromDisk<RayCaster::Vol>(vol_3d_file);

    vout << "creating spec: " << spec_ID << " in dst_h5_file" << std::endl;
    H5::Group dst_spec_g = dst_h5.createGroup(spec_ID);

    vout << "adding volume to HDF5..." << std::endl;
    H5::Group dst_spec_vol_group = dst_spec_g.createGroup("vol");
    WriteImageH5(vol_hu.GetPointer(), &dst_spec_vol_group);
  }

  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
