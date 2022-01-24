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

int main(int argc, char* argv[])
{
  using namespace xreg;

  constexpr int kEXIT_VAL_SUCCESS = 0;
  constexpr int kEXIT_VAL_BAD_USE = 1;

  // First, set up the program options

  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Generate IPCAI data groundtruth h5 file by projecting each specimen CT"
              "to lands and segs using groundtruth pelvis pose.");

  po.set_arg_usage("<IPCAI full res H5 File> <Dst H5 File> <3D Landmarks>"
                   "<xreg output path> <real output path> <debug output path>");
  po.set_min_num_pos_args(4);
  
  po.add("preproc", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "preproc",
         "Apply standard preprocessing such as cropping and, if the \"intensity\" flag is set, "
         "log intensity conversion.")
    << false;

  po.add("ds-factor", 'd', ProgOpts::kSTORE_DOUBLE, "ds-factor",
         "Downsample the camera (2D image) dimensions by this factor, this is done before "
         "projection. e.g. 0.5 --> shrink ny half, 1.0 --> no downsampling.")
    << 1.0;

  po.add("proj-idx", 'p', ProgOpts::kSTORE_UINT32, "proj-idx",
         "Index of the projection to project to (the projection data file may store several projections)")
    << ProgOpts::uint32(0);

  po.add("intensity", 'i', ProgOpts::kSTORE_STRING, "intensity",
         "Save a copy of the intensity projection data at this path, after cropping, downsampling, etc.")
    << "";

  po.add("no-ras2lps", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "no-ras2lps",
         "Do NOT convert RAS to LPS (or LPS to RAS) for the 3D landmarks; "
         "RAS to LPS conversion negates the first and second components.")
    << false;
  po.add("write-debug-files", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "write-debug-files",
         "Write debug files to disk and projs to dst h5 file, default is False")
    << false;
  po.add("write-img-to-disk", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "write-img-to-disk",
         "Write image files (xreg & real) to disk")
    << false;
  po.add("real-label", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "real-label",
         "Real label enables left and right femur groundtruth pose. IF false, they are pelvis pose")
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

  const std::string ipcai_ful_res_h5_path   = po.pos_args()[0];
  const std::string dst_h5_path             = po.pos_args()[1];
  const std::string fcsv_3d_path            = po.pos_args()[2];
  const std::string xreg_output_path        = po.pos_args()[3];
  const std::string real_output_path        = po.pos_args()[4];
  const std::string debug_output_path       = po.pos_args()[5];

  // const std::string multi_chan_2d_seg_path = (po.pos_args().size() > 7) ? po.pos_args()[7] : std::string();

  const double ds_factor = po.get("ds-factor");

  const bool do_ds = std::abs(ds_factor - 1.0) > 0.001;

  const bool ras2lps = !po.get("no-ras2lps").as_bool();

  const size_type proj_idx = po.get("proj-idx").as_uint32();

  const std::string dst_intens_proj_path = po.get("intensity");

  const bool save_intens_proj = !dst_intens_proj_path.empty();

  const bool apply_preproc = po.get("preproc");

  const bool write_debug_files = po.get("write-debug-files");

  const bool write_img_to_disk = po.get("write-img-to-disk");

  const bool real_label = po.get("real-label");

  vout << "opening output file for writing..." << std::endl;
  H5::H5File dst_h5(dst_h5_path, H5F_ACC_TRUNC);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // ** Write proj-params **
  //
  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  // map world --> camera 3D
  const FrameTransform extrins = default_cam.extrins;

  // project camera 3D --> image plane 2D pixels
  const Mat3x3 intrins = default_cam.intrins;

  // Every projection should have the same extrinsics/intrinsics for this dataset
  vout << "writing proj-params ..." << std::endl;
  {
    H5::Group cam_g = dst_h5.createGroup("proj-params");

    WriteSingleScalarH5("num-rows", default_cam.num_det_rows, &cam_g);
    WriteSingleScalarH5("num-cols", default_cam.num_det_cols, &cam_g);

    WriteSingleScalarH5("pixel-row-spacing", default_cam.det_row_spacing, &cam_g);
    WriteSingleScalarH5("pixel-col-spacing", default_cam.det_col_spacing, &cam_g);

    WriteMatrixH5("intrinsic", intrins, &cam_g);
    WriteAffineTransform4x4("extrinsic", extrins, &cam_g);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // ** Write land-names **
  //
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

  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  // ** Write Each Specimen **
  //
  std::vector<std::string> spec_ID_list = {"17-1882", "18-1109", "18-0725", "18-2799", "18-2800", "17-1905"};
  std::vector<std::string> spec_num_list = {"01", "02", "03", "04", "05", "06"};
  std::vector<size_type> spec_projs_num_list = {111, 104, 24, 48, 55, 24};
  const size_type num_specs = 6;

  H5::H5File h5(ipcai_ful_res_h5_path, H5F_ACC_RDWR);

  for(size_type spec_idx = 0; spec_idx < num_specs; ++spec_idx)
  {
    const std::string spec_ID = spec_ID_list[spec_idx];
    const std::string spec_num = spec_num_list[spec_idx];
    vout << "creating spec: " << spec_num << " in dst_h5_file" << std::endl;
    H5::Group dst_spec_g = dst_h5.createGroup(spec_num);
    H5::Group gt_poses_g = dst_spec_g.createGroup("gt-poses");

    bool need_to_init_ds = true;

    H5::Group ipcai_spec_ID_g = h5.openGroup(spec_ID);
    vout << "reading input CT volume..." << std::endl;
    H5::Group ipcai_spec_ID_vol_g = ipcai_spec_ID_g.openGroup("vol");
    auto vol_hu = ReadITKImageH5Float3D(ipcai_spec_ID_vol_g);

    vout << "converting HU --> Lin. Att." << std::endl;
    auto vol_intens = HUToLinAtt(vol_hu.GetPointer(), -130);

    vout << "reading label map..." << std::endl;
    H5::Group ipcai_spec_ID_seg_g = ipcai_spec_ID_g.openGroup("vol-seg/image");
    auto ct_labels = ReadITKImageH5UChar3D(ipcai_spec_ID_seg_g);

    vout << "reading 3D landmarks..." << std::endl;
    const std::string lands_3d_file = fcsv_3d_path + '/' + spec_ID + "_landmark.fcsv";
    LandMap3 lands_3d = ReadFCSVFileNamePtMap(lands_3d_file);

    if (ras2lps)
    {
      vout << "converting landmarks from RAS -> LPS" << std::endl;
      ConvertRASToLPS(&lands_3d);
    }

    vout << "3D Landmarks:\n";
    PrintLandmarkMap(lands_3d, vout);

    /*
        vout << "adding volume to HDF5..." << std::endl;
        H5::Group dst_spec_vol_group = dst_spec_g.createGroup("vol");
        WriteImageH5(vol_hu.GetPointer(), &dst_spec_vol_group);

        vout << "reading projection data..." << std::endl;
        DeferredProjReader pd_reader(proj_data_path);

        xregASSERT(proj_idx < pd_reader.num_projs_on_disk());

        auto pd = pd_reader.proj_data_F32()[proj_idx];

        if (save_intens_proj)
        {
          pd.img = pd_reader.read_proj_F32(proj_idx);
        }
    */

    H5::Group ipcai_spec_ID_projections_g = ipcai_spec_ID_g.openGroup("projections");
    size_type num_projs = spec_projs_num_list[spec_idx];

    H5::DataSet proj_ds;
    H5::DataSet seg_ds;
    H5::DataSet lands_ds;

    using LandMatRef = Eigen::Map<Eigen::Matrix<CoordScalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor|Eigen::DontAlign>>;

    std::vector<CoordScalar> land_buf(proj_num_lands * 2);
    LandMatRef lands_buf_mat_ref(&land_buf[0], 2, proj_num_lands);

    for(size_type proj_idx = 0; proj_idx < num_projs; ++proj_idx)
    {
      std::cout << "running spec: " << spec_ID << " proj idx: " << proj_idx << std::endl;

      const std::string proj_idx_fmt = fmt::format("{:03d}", proj_idx);

      H5::Group ipcai_proj_g = ipcai_spec_ID_projections_g.openGroup(proj_idx_fmt);
      H5::Group ipcai_gt_poses_g = ipcai_proj_g.openGroup("gt-poses");
      const FrameTransform pelvis_pose = ReadAffineTransform4x4H5("cam-to-pelvis-vol", ipcai_gt_poses_g);
      const FrameTransform left_femur_pose = real_label ? ReadAffineTransform4x4H5("cam-to-left-femur-vol", ipcai_gt_poses_g) : pelvis_pose;
      const FrameTransform right_femur_pose = real_label ? ReadAffineTransform4x4H5("cam-to-right-femur-vol", ipcai_gt_poses_g) : pelvis_pose;

      WriteAffineTransform4x4(proj_idx_fmt, pelvis_pose, &gt_poses_g);

      auto rot_pat_up = ReadSingleScalarH5Bool("rot-180-for-up", ipcai_proj_g);

      H5::Group ipcai_proj_image_g = ipcai_proj_g.openGroup("image");
      auto real_proj = ReadITKImageH5Float2D(ipcai_proj_image_g);

      vout << "  ray casting..." << std::endl;
      auto rc = LineIntRayCasterFromProgOpts(po);
      rc->set_camera_models(RayCaster::CameraModelList(1,default_cam));
      rc->set_num_projs(1);
      rc->set_volume(vol_intens);

      rc->allocate_resources();
      rc->distribute_xform_among_cam_models(pelvis_pose);
      rc->compute();

      ProjDataF32 pd;

      const size_type num_photons = 2000;

      pd.cam = default_cam;
      //pd.img = SamplePoissonProjFromAttProj(rc->proj(0).GetPointer(), num_photons);
      pd.img = rc->proj(0).GetPointer();

      pd.rot_to_pat_up = rot_pat_up ? ProjDataRotToPatUp::kONE_EIGHTY : ProjDataRotToPatUp::kZERO;

      if (apply_preproc)
      {
        vout << "projection preprocessing..." << std::endl;

        ProjPreProc preproc;
        preproc.set_debug_output_stream(vout, verbose);

        preproc.input_projs = { pd };

        preproc();

        pd = preproc.output_projs[0];
      }

      ProjDataF32 pd_real;

      pd_real.cam = default_cam;
      //pd.img = SamplePoissonProjFromAttProj(rc->proj(0).GetPointer(), num_photons);
      pd_real.img = real_proj.GetPointer();

      pd_real.rot_to_pat_up = rot_pat_up ? ProjDataRotToPatUp::kONE_EIGHTY : ProjDataRotToPatUp::kZERO;

      if (apply_preproc)
      {
        vout << "projection preprocessing..." << std::endl;

        ProjPreProc preproc;
        preproc.set_debug_output_stream(vout, verbose);

        preproc.input_projs = { pd_real };

        preproc();

        pd_real = preproc.output_projs[0];
      }

      if (do_ds)
      {
        vout << "downsampling input projection data..." << std::endl;
        pd = DownsampleProjData(pd, ds_factor);
        pd_real = DownsampleProjData(pd_real, ds_factor);
      }

      const auto& cam = pd.cam;

      vout << "projecting 3D labels to 2D..." << std::endl;
      auto proj_labels = ProjectHipLabels(ct_labels, cam, pelvis_pose, left_femur_pose, right_femur_pose,
                                          ProgOptsDepthRayCasterFactory(po), verbose, vout);

      vout << "creating proj. data. struct for the single-channel segmentation..."  << std::endl;
      ProjDataU8 seg_pd;
      {
        const FrameTransform vol_to_cam = pelvis_pose.inverse();

        seg_pd.cam = cam;
        seg_pd.img = std::get<0>(proj_labels);

        for (const auto& lkv : lands_3d)
        {
          seg_pd.landmarks.emplace(lkv.first, cam.phys_pt_to_ind_pt(vol_to_cam * lkv.second).head(2));
        }
      }

      if (rot_pat_up && pd.rot_to_pat_up && (*pd.rot_to_pat_up != ProjDataRotToPatUp::kZERO))
      {
        xregASSERT(*pd.rot_to_pat_up == ProjDataRotToPatUp::kONE_EIGHTY);

        std::cout << "rotating projections for patient up..." << std::endl;

        ModifyForPatUp(seg_pd.img.GetPointer(), *pd.rot_to_pat_up);

        ModifyForPatUp(pd.img.GetPointer(), *pd.rot_to_pat_up);

        ModifyForPatUp(pd_real.img.GetPointer(), *pd_real.rot_to_pat_up);

        pd.rot_to_pat_up.reset();

        for (auto& lkv : seg_pd.landmarks)
        {
          lkv.second[0] = static_cast<CoordScalar>(cam.num_det_cols - 1) - lkv.second[0];
          lkv.second[1] = static_cast<CoordScalar>(cam.num_det_rows - 1) - lkv.second[1];
        }

        if (save_intens_proj)
        {
          ModifyForPatUp(pd.img.GetPointer(), *pd.rot_to_pat_up);
          pd.rot_to_pat_up.reset();
        }
      }
      else
      {
        seg_pd.rot_to_pat_up = pd.rot_to_pat_up;
      }

      if (save_intens_proj)
      {
        pd.landmarks = seg_pd.landmarks;

        vout << "writing processed intensity proj. data to disk..." << std::endl;
        WriteProjDataH5ToDisk(pd, dst_intens_proj_path);
      }

      // ** Writing to disk ** //////////////////////////////////////////////////////////////////////////////////////
      if(write_img_to_disk)
      {
        vout << "writing xreg DRR to disk..." << std::endl;
        const std::string xreg_proj_tiff_path = xreg_output_path + "/" + spec_num + "/" + proj_idx_fmt + ".tiff";
        WriteITKImageToDisk(pd.img.GetPointer(), xreg_proj_tiff_path);

        vout << "writing real Xray to disk..." << std::endl;
        const std::string real_proj_tiff_path = real_output_path + "/" + spec_num + "/" + proj_idx_fmt + ".tiff";
        WriteITKImageToDisk(pd_real.img.GetPointer(), real_proj_tiff_path);
      }

      if(write_debug_files)
      {
        vout << "writing png seg image object to disk..." << std::endl;
        const std::string seg_png_path = debug_output_path + "/" + spec_num + "_seg_" + std::to_string(proj_idx) + ".png";
        WriteITKImageRemap8bpp(seg_pd.img.GetPointer(), seg_png_path);

        vout << "writing png proj image object to disk..." << std::endl;
        const std::string proj_png_path = debug_output_path + "/" + spec_num + "_proj_" + std::to_string(proj_idx) + ".png";
        WriteITKImageRemap8bpp(pd.img.GetPointer(), proj_png_path);

        const std::string proj_disk_path = debug_output_path + "/" + spec_num + "_proj_" + std::to_string(proj_idx) + ".h5";
        WriteProjDataH5ToDisk(pd, proj_disk_path);
      }

      const size_type proj_num_cols = pd.cam.num_det_cols;
      const size_type proj_num_rows = pd.cam.num_det_rows;

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

          if(write_debug_files)
          {
            vout << "        creating projs dataset..." << std::endl;
            proj_ds = dst_spec_g.createDataSet("projs", LookupH5DataType<RayCaster::PixelScalar2D>(), data_space, props);
          }

          vout << "        creating segs dataset..." << std::endl;
          seg_ds = dst_spec_g.createDataSet("segs", LookupH5DataType<unsigned char>(), data_space, props);
        }

        // Create lands dataset
        {
          const std::array<hsize_t,3> lands_ds_dims = { num_projs, 2, proj_num_lands };

          H5::DataSpace data_space(lands_ds_dims.size(), lands_ds_dims.data());

          // chunks for compression
          H5::DSetCreatPropList props;
          props.copy(H5::DSetCreatPropList::DEFAULT);

          // chunk the entire array
          std::array<hsize_t,3> chunk_dims = { num_projs, 2, proj_num_lands };
          props.setChunk(chunk_dims.size(), chunk_dims.data());
          props.setDeflate(9);

          vout << "      creating lands dataset..." << std::endl;
          lands_ds = dst_spec_g.createDataSet("lands", LookupH5DataType<CoordScalar>(), data_space, props);
        }

        need_to_init_ds = false;
      }

      {
        const std::array<hsize_t,3> m_dims = { 1, proj_num_rows, proj_num_cols };
        H5::DataSpace ds_m(m_dims.size(), m_dims.data());

        const std::array<hsize_t,3> f_start = { proj_idx, 0, 0 };

        if(write_debug_files)
        {
          vout << "      writing projection data..." << std::endl;

          H5::DataSpace ds_f = proj_ds.getSpace();
          ds_f.selectHyperslab(H5S_SELECT_SET, m_dims.data(), f_start.data());

          proj_ds.write(pd.img->GetBufferPointer(), LookupH5DataType<RayCaster::PixelScalar2D>(), ds_m, ds_f);
        }

        {
          vout << "      writing segmentation data..." << std::endl;

          H5::DataSpace ds_f = seg_ds.getSpace();
          ds_f.selectHyperslab(H5S_SELECT_SET, m_dims.data(), f_start.data());

          seg_ds.write(seg_pd.img->GetBufferPointer(), LookupH5DataType<unsigned char>(), ds_m, ds_f);
        }

      }

      {
        vout << "      writing landmark data..." << std::endl;

        for (size_type land_idx = 0; land_idx < proj_num_lands; ++land_idx)
        {
          const auto cur_land_it = seg_pd.landmarks.find(land_names[land_idx]);
          xregASSERT(cur_land_it != seg_pd.landmarks.end());

          lands_buf_mat_ref(0,land_idx) = cur_land_it->second(0);
          lands_buf_mat_ref(1,land_idx) = cur_land_it->second(1);
        }

        const std::array<hsize_t,3> m_dims = { 1, 2, proj_num_lands };
        H5::DataSpace ds_m(m_dims.size(), m_dims.data());

        const std::array<hsize_t,3> f_start = { proj_idx, 0, 0 };

        H5::DataSpace ds_f = lands_ds.getSpace();
        ds_f.selectHyperslab(H5S_SELECT_SET, m_dims.data(), f_start.data());

        // recall that lands_buf_mat_ref uses land_buf
        lands_ds.write(&land_buf[0], LookupH5DataType<CoordScalar>(), ds_m, ds_f);
      }
    }
  }

  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
