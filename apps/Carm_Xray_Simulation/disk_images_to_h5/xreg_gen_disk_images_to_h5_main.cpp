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

  po.set_help("Embed disk images to h5 file"
              "Images are sorted in the order of <root folder>/<specID>/000");

  po.set_arg_usage("< Dst H5 File > < Images root folder path > < File prefix (eg: DeepDRR)>"
                   "< SpecID 1 > < Image Num 1 > < SpecID 2 > < Image Num 2 >... < SpecID N > < Image Num N >");
  po.set_min_num_pos_args(4);

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

  po.add("num-len", 'n', ProgOpts::kSTORE_STRING, "num-len",
         "Length of projection index, for example 3 is 000.")
    << "3";
  po.add("img-write-path", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_STRING, "img-write-path",
         "Path that the images write to.")
    << "";

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

  const std::string dst_h5_path      = po.pos_args()[0];
  const std::string image_path       = po.pos_args()[1];

  std::vector<std::string> spec_ID_list(po.pos_args().begin() + 2, po.pos_args().end());

  const bool do_logxform = po.get("logxform");
  const double ds_factor = po.get("ds-factor");
  const bool do_ds = std::abs(ds_factor - 1.0) > 0.001;
  const size_type num_specs = spec_ID_list.size();
  const std::string num_len = po.get("num-len");
  const size_type img_size = po.get("img-size").as_uint32();
  const bool apply_preproc = po.get("preproc");
  const std::string image_write_path = po.get("img-write-path");

  xregASSERT(img_size > 0);

  vout << "opening output file for writing..." << std::endl;
  H5::H5File dst_h5(dst_h5_path, H5F_ACC_TRUNC);

  const std::vector<std::string> land_names = { "FH-l",   "FH-r",
                                                "GSN-l",  "GSN-r",
                                                "IOF-l",  "IOF-r",
                                                "MOF-l",  "MOF-r",
                                                "SPS-l",  "SPS-r",
                                                "IPS-l",  "IPS-r",
                                                "ASIS-l", "ASIS-r" };

  const size_type proj_num_lands = land_names.size();

  vout << "writing landmark names..." << std::endl;
  {
    H5::Group dst_land_names_g = dst_h5.createGroup("land-names");

    WriteSingleScalarH5("num-lands", proj_num_lands, &dst_land_names_g);

    for (size_type land_idx = 0; land_idx < proj_num_lands; ++land_idx)
    {
      WriteStringH5(fmt::format("land-{:02d}", land_idx), land_names[land_idx], &dst_land_names_g);
    }
  }

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  for(size_type spec_idx = 0; spec_idx < num_specs; spec_idx += 2)
  {
    bool need_to_init_ds = true;

    const std::string spec_ID = spec_ID_list[spec_idx];

    vout << "creating spec: " << spec_ID << " in dst_h5_file" << std::endl;
    H5::Group dst_spec_g = dst_h5.createGroup(spec_ID);

    H5::DataSet proj_ds;

    size_type num_projs = std::stoi(spec_ID_list[spec_idx+1]);

    for(size_type proj_idx = 0; proj_idx < num_projs; ++proj_idx)
    {
      ProjDataF32 pd;

      const std::string img_file = image_path + "/" + spec_ID + "/" + fmt::format("{:0" + num_len + "d}", proj_idx) + ".tiff";
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

      if (image_write_path != "")
      {
        vout << "writing img to disk..." << std::endl;
        const std::string proj_tiff_path = image_write_path + "/" + spec_ID + "/" + fmt::format("{:0" + num_len + "d}", proj_idx) + ".tiff";
        WriteITKImageToDisk(pd.img.GetPointer(), proj_tiff_path);
      }

      const size_type proj_num_cols = img_size;
      const size_type proj_num_rows = img_size;

      if (need_to_init_ds)
      {
        // Create projs & segs dataset
        {
          const std::array<hsize_t,3> proj_ds_dims = { num_projs, proj_num_rows, proj_num_cols };

          H5::DataSpace data_space(proj_ds_dims.size(), proj_ds_dims.data());

          // chunks for compression
          H5::DSetCreatPropList props;
          props.copy(H5::DSetCreatPropList::DEFAULT);

          // chunk of 1 image
          std::array<hsize_t,3> chunk_dims = { 1, proj_num_rows, proj_num_cols };
          props.setChunk(chunk_dims.size(), chunk_dims.data());
          props.setDeflate(9);

          vout << "        creating projs dataset..." << std::endl;
          proj_ds = dst_spec_g.createDataSet("projs", LookupH5DataType<RayCaster::PixelScalar2D>(), data_space, props);
        }

        need_to_init_ds = false;
      }

      {
        const std::array<hsize_t,3> m_dims = { 1, proj_num_rows, proj_num_cols };
        H5::DataSpace ds_m(m_dims.size(), m_dims.data());

        const std::array<hsize_t,3> f_start = { proj_idx, 0, 0 };

        {
          vout << "      writing projection data..." << std::endl;

          H5::DataSpace ds_f = proj_ds.getSpace();
          ds_f.selectHyperslab(H5S_SELECT_SET, m_dims.data(), f_start.data());

          proj_ds.write(pd.img->GetBufferPointer(), LookupH5DataType<RayCaster::PixelScalar2D>(), ds_m, ds_f);
        }

      }
    }
  }

  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
