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

#include "IPCAICommon.h"

int main(int argc, char* argv[])
{
  using namespace xreg;

  constexpr int kEXIT_VAL_SUCCESS = 0;
  constexpr int kEXIT_VAL_BAD_USE = 1;

  // First, set up the program options

  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Embed disk images to h5 file"
              "Images are sorted in the order of <root folder>/<specID>/000");

  po.set_arg_usage("< Dst H5 File > < Images root folder path > < File prefix (eg: DeepDRR)>"
                   "< SpecID 1 > < Image Num 1 > < SpecID 2 > < Image Num 2 >... < SpecID N > < Image Num N >");
  po.set_min_num_pos_args(3);
  po.set_help_epilogue(fmt::format("\nIPCAI version: {}", IPCAIVersionStr()));

  po.add("logxform", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "logxform",
         "log intensity conversion.")
    << false;

  po.add("preproc", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "preproc",
         "Apply standard preprocessing such as cropping and, if the \"intensity\" flag is set, "
         "log intensity conversion.")
    << false;

  po.add("ds-factor", 'd', ProgOpts::kSTORE_DOUBLE, "ds-factor",
         "Downsample the camera (2D image) dimensions by this factor, this is done before "
         "projection. e.g. 0.5 --> shrink ny half, 1.0 --> no downsampling.")
    << 1.0;

  po.add("img-size", 's', ProgOpts::kSTORE_UINT32, "img-size",
         "row(or col) of the image, for example: 180")
    << ProgOpts::uint32(0);

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

  const std::string image_ID     = po.pos_args()[0];
  const std::string image_path   = po.pos_args()[1];
  const std::string write_path   = po.pos_args()[2];

  std::vector<std::string> spec_ID_list(po.pos_args().begin() + 2, po.pos_args().end());

  const bool do_logxform = po.get("logxform");
  const double ds_factor = po.get("ds-factor");
  const bool do_ds = std::abs(ds_factor - 1.0) > 0.001;
  const size_type img_size = po.get("img-size").as_uint32();
  const bool apply_preproc = po.get("preproc");

  xregASSERT(img_size > 0);

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  ProjDataF32 pd;

  const std::string img_file = image_path + "/" + image_ID;
  vout << "reading image..." << img_file << "\n";
  pd.cam = default_cam;
  pd.img = ReadITKImageFromDisk<itk::Image<float,2>>(img_file);

  const auto& cam = pd.cam;

  if (apply_preproc)
  {
    vout << "projection preprocessing..." << std::endl;

    ProjPreProc preproc;
    preproc.set_debug_output_stream(vout, verbose);

    preproc.input_projs = { pd };

    preproc();

    pd = preproc.output_projs[0];
  }

  if (do_logxform)
  {
    vout << "Log transforming..." << std::endl;

    auto log_xform = ImageIntensLogTransFilter::New();
    log_xform->SetInput(pd.img);
    log_xform->SetUseMaxIntensityAsI0(true);
    log_xform->Update();

    pd.img = log_xform->GetOutput();
  }

  if (do_ds)
  {
    vout << "downsampling input projection data..." << std::endl;
    pd = DownsampleProjData(pd, ds_factor);
  }

  vout << "writing img to disk..." << std::endl;
  const std::string proj_tiff_path = write_path + "/" + image_ID + ".tiff";
  WriteITKImageToDisk(pd.img.GetPointer(), proj_tiff_path);

  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
