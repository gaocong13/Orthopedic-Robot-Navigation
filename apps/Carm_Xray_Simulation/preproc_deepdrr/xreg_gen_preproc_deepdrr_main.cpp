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
#include "xregImageIntensLogTrans.h"

int main(int argc, char* argv[])
{
  using namespace xreg;

  constexpr int kEXIT_VAL_SUCCESS = 0;
  constexpr int kEXIT_VAL_BAD_USE = 1;

  // First, set up the program options

  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Log xform DeepDRR images");

  po.set_arg_usage("< input image path > < output path >");
  po.set_min_num_pos_args(2);

  po.add("logxform", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "logxform",
         "log intensity conversion.")
    << true;

  po.add("img-size", 's', ProgOpts::kSTORE_UINT32, "img-size",
         "row(or col) of the image, for example: 180")
    << ProgOpts::uint32(0);

  po.add("num-img", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_UINT32, "num-img",
         "number of images to convert")
    << ProgOpts::uint32(0);

  po.add("num-len", 'n', ProgOpts::kSTORE_STRING, "num-len",
         "Length of projection index, for example 3 is 000.")
    << "5";

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

  const std::string image_path      = po.pos_args()[0];
  const std::string output_path     = po.pos_args()[1];

  const bool do_logxform = po.get("logxform");
  const std::string num_len = po.get("num-len");
  const size_type img_size = po.get("img-size").as_uint32();
  const size_type num_projs = po.get("num-img").as_uint32();

  xregASSERT(img_size > 0);
  xregASSERT(num_projs > 0);

  size_type count_idx = 0;

  for(size_type proj_idx = 0; proj_idx < num_projs; ++proj_idx)
  {
    const std::string img_file = image_path + "/" + fmt::format("{:0" + num_len + "d}", proj_idx) + ".tiff";
    vout << "reading image..." << img_file << "\n";
    auto img = ReadITKImageFromDisk<itk::Image<float,2>>(img_file);

    if (do_logxform)
    {
      vout << "Log transforming..." << std::endl;

      auto log_xform = ImageIntensLogTransFilter::New();
      log_xform->SetInput(img);
      log_xform->SetUseMaxIntensityAsI0(true);
      log_xform->Update();

      img = log_xform->GetOutput();
    }

    {
      vout << "writing img to disk..." << std::endl;
      const std::string proj_tiff_path = output_path + "/" + fmt::format("{:0" + num_len + "d}", count_idx) + ".tiff";
      WriteITKImageToDisk(img.GetPointer(), proj_tiff_path);
      ++count_idx;
    }
  }

  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
