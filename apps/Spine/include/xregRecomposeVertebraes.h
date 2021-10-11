//
//  main.cpp
//  TestITK
//
//  Created by Cong Gao on 4/2/18.
//  Copyright © 2018 Cong Gao. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include "itkCenteredAffineTransform.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTileImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkAddImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkWindowedSincInterpolateImageFunction.h"
#include "itkImage.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "omp.h"
#include "assert.h"

#include "itkAffineTransform.h"
#include <itkEuler3DTransform.h>

#include "xregITKIOUtils.h"
#include "xregITKLabelUtils.h"
#include "xregHUToLinAtt.h"

// #include "spline.h"

const unsigned int Dimension = 3;
constexpr unsigned int Radius = 3;
const float PI = 3.14159;

namespace xreg
{
  typedef float  InputPixelType;
  typedef float  OutputPixelType;
  typedef float  ScalarType;

  using Vol         = itk::Image<InputPixelType,3>;
  using VolPtr      = typename Vol::Pointer;

  typedef itk::ImageFileReader< Vol  >  ReaderType;
  typedef itk::ImageFileWriter< Vol >  WriterType;
  typedef itk::AddImageFilter< Vol, Vol > AddFilterType;
  typedef itk::ResampleImageFilter< Vol, Vol, ScalarType, ScalarType > ResampleFilterType;

  using InterpolatorType = itk::WindowedSincInterpolateImageFunction< Vol, Radius >;
  using TransformType = itk::AffineTransform< ScalarType, Dimension >;
  using MatrixType = itk::Matrix<ScalarType, Dimension + 1, Dimension + 1>;
  using EulerTransformType = itk::Euler3DTransform< ScalarType >;

  class RecomposeVertebrae
  {
    public:
      RecomposeVertebrae(const std::string meta_data_path,
                         const float rot_angX_rand,
                         const float rot_angY_rand,
                         const float rot_angZ_rand,
                         const bool save_hu = false);


      VolPtr ResampleVertebrae(std::ostream& vout);
      FrameTransform GetVert1Xform();
      FrameTransform GetVert2Xform();
      FrameTransform GetVert3Xform();
      FrameTransform GetVert4Xform();

    private:
      void AddVolumes(AddFilterType::Pointer addFilter, VolPtr inputImage1, VolPtr inputImage2);
      void ResampleVolume(ResampleFilterType::Pointer resample, VolPtr inputImage, EulerTransformType::Pointer transform);
      Pt3 ReadVertbRotCenter(LandMap3 lands_fcsv, const std::string vert_name);
      void TransformVertebrae(EulerTransformType::Pointer vert_xform,
                                              ScalarType rot_angX,
                                              ScalarType rot_angY,
                                              ScalarType rot_angZ,
                                              ScalarType transX,
                                              ScalarType transY,
                                              ScalarType transZ,
                                              Pt3 vert_rot_cen);
      FrameTransform ComputeVertbAffineTransform(EulerTransformType::Pointer vert_xform,
                                                                     ScalarType transX,
                                                                     ScalarType transY,
                                                                     ScalarType transZ);
      float _rot_angX_rand;
      float _rot_angY_rand;
      float _rot_angZ_rand;
      std::string _meta_data_path;
      FrameTransform _vert1_xform4x4;
      FrameTransform _vert2_xform4x4;
      FrameTransform _vert3_xform4x4;
      FrameTransform _vert4_xform4x4;
      bool _save_hu;
  };

  RecomposeVertebrae::RecomposeVertebrae(const std::string meta_data_path,
                                         const float rot_angX_rand,
                                         const float rot_angY_rand,
                                         const float rot_angZ_rand,
                                         const bool save_hu)
  {
    _rot_angX_rand = rot_angX_rand;
    _rot_angY_rand = rot_angY_rand;
    _rot_angZ_rand = rot_angZ_rand;
    _meta_data_path = meta_data_path;
    _vert1_xform4x4 = FrameTransform::Identity();
    _vert2_xform4x4 = FrameTransform::Identity();
    _vert3_xform4x4 = FrameTransform::Identity();
    _vert4_xform4x4 = FrameTransform::Identity();
    _save_hu = save_hu;
  }

  FrameTransform RecomposeVertebrae::GetVert1Xform()
  {
    return _vert1_xform4x4;
  }

  FrameTransform RecomposeVertebrae::GetVert2Xform()
  {
    return _vert2_xform4x4;
  }

  FrameTransform RecomposeVertebrae::GetVert3Xform()
  {
    return _vert3_xform4x4;
  }

  FrameTransform RecomposeVertebrae::GetVert4Xform()
  {
    return _vert4_xform4x4;
  }

  FrameTransform RecomposeVertebrae::ComputeVertbAffineTransform(EulerTransformType::Pointer vert_xform,
                                                                 ScalarType transX,
                                                                 ScalarType transY,
                                                                 ScalarType transZ)
  {
    FrameTransform vert_xform4x4;
    auto rot_mat = vert_xform->GetMatrix();
    for(size_type i = 0; i < 3; ++i)
      for(size_type j = 0; j < 3; ++j)
      {
        vert_xform4x4(i, j) = rot_mat[i][j];
      }
    vert_xform4x4(0, 3) = transX;
    vert_xform4x4(1, 3) = transY;
    vert_xform4x4(2, 3) = transZ;

    return vert_xform4x4;
  }

  void RecomposeVertebrae::AddVolumes(AddFilterType::Pointer addFilter,
                                                      VolPtr inputImage1,
                                                      VolPtr inputImage2)
  {
      addFilter->SetInput( 0, inputImage1 );
      addFilter->SetInput( 1, inputImage2 );
      addFilter->Update();
  }

  void RecomposeVertebrae::ResampleVolume(ResampleFilterType::Pointer resample,
                                                               VolPtr inputImage,
                                          EulerTransformType::Pointer transform)
  {
    const Vol::SizeType & size = inputImage->GetLargestPossibleRegion().GetSize();
    resample->SetInput(inputImage);
    resample->SetReferenceImage(inputImage);
    resample->UseReferenceImageOn();
    resample->SetSize(size);
    resample->SetDefaultPixelValue(0);

    // InterpolatorType::Pointer interpolator = InterpolatorType::New();
    // resample->SetInterpolator(interpolator);

    // get transform parameters from MatrixType
    /*
    TransformType::Pointer transform = TransformType::New();
    TransformType::ParametersType parameters(Dimension * Dimension + Dimension);
    for (unsigned int i = 0; i < Dimension; i++)
    {
      for (unsigned int j = 0; j < Dimension; j++)
      {
        parameters[i * Dimension + j] = matrix[i][j];
      }
    }
    for (unsigned int i = 0; i < Dimension; i++)
    {
      parameters[i + Dimension * Dimension] = matrix[i][Dimension];
    }
    transform->SetParameters(parameters);
    */
    resample->SetTransform(transform);
  }

  std::ifstream& GotoLine(std::ifstream& file, unsigned int num){
      file.seekg(std::ios::beg);
      for(int i=0; i < num - 1; ++i){
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
      }
      return file;
  }

  Pt3 RecomposeVertebrae::ReadVertbRotCenter(LandMap3 lands_fcsv, const std::string vert_name)
  {
    Pt3 rotcenter;

    auto lands_fcsv_rotc = lands_fcsv.find(vert_name);
    if (lands_fcsv_rotc != lands_fcsv.end()){
      rotcenter = lands_fcsv_rotc->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND " << vert_name << std::endl;
    }

    return rotcenter;
  }

  void RecomposeVertebrae::TransformVertebrae(EulerTransformType::Pointer vert_xform,
                                                               ScalarType rot_angX,
                                                               ScalarType rot_angY,
                                                               ScalarType rot_angZ,
                                                               ScalarType transX,
                                                               ScalarType transY,
                                                               ScalarType transZ,
                                                               Pt3 vert_rot_cen)
  {
    EulerTransformType::ParametersType vert_parameters(6);
    vert_parameters[0] = rot_angX * kDEG2RAD;
    vert_parameters[1] = rot_angY * kDEG2RAD;
    vert_parameters[2] = rot_angZ * kDEG2RAD;
    vert_parameters[3] = transX;
    vert_parameters[4] = transY;
    vert_parameters[5] = transZ;

    vert_xform->SetParameters(vert_parameters);

    EulerTransformType::FixedParametersType vert_fixedparameters(3);
    vert_fixedparameters[0] = vert_rot_cen[0];
    vert_fixedparameters[1] = vert_rot_cen[1];
    vert_fixedparameters[2] = vert_rot_cen[2];
    vert_xform->SetFixedParameters(vert_fixedparameters);
  }

  VolPtr RecomposeVertebrae::ResampleVertebrae(std::ostream& vout)
  {
    const std::string spinevol_path = _meta_data_path + "/Spine21-2512_CT_crop.nrrd";
    const std::string spineseg_path = _meta_data_path + "/Spine21-2512_seg_crop.nrrd";
    const std::string sacrumseg_path = _meta_data_path + "/Spine21-2512_sacrum_seg_crop.nrrd";
    const std::string spine_3d_fcsv_path = _meta_data_path + "/Spine_3D_landmarks.fcsv";
    const std::string vert_rotfcsv_path = _meta_data_path + "/vertb-rot.fcsv";

    vout << "  [ResampleVertebrae] - reading spine anatomical landmarks from FCSV file..." << std::endl;

    auto spine_3d_fcsv = ReadFCSVFileNamePtMap(spine_3d_fcsv_path);
    ConvertRASToLPS(&spine_3d_fcsv);

    vout << "  [ResampleVertebrae] - reading vertebrae rotation centers from FCSV file..." << std::endl;
    auto vert_rot_fcsv = ReadFCSVFileNamePtMap(vert_rotfcsv_path);
    ConvertRASToLPS(&vert_rot_fcsv);

    const bool use_seg = true;
    auto spine_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(spineseg_path);

    auto sacrum_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(sacrumseg_path);

    vout << "  [ResampleVertebrae] - reading spine volume..." << std::endl; // We only use the needle metal part
    auto spinevol_hu = ReadITKImageFromDisk<Vol>(spinevol_path);

    {
      spinevol_hu->SetOrigin(spine_seg->GetOrigin());
      spinevol_hu->SetSpacing(spine_seg->GetSpacing());
    }

    vout << "  [ResampleVertebrae] - HU --> Att. ..." << std::endl;
    auto spinevol_att = _save_hu ? spinevol_hu : HUToLinAtt(spinevol_hu.GetPointer());

    {
      sacrum_seg->SetOrigin(spinevol_hu->GetOrigin());
      sacrum_seg->SetSpacing(spinevol_hu->GetSpacing());
    }

    const unsigned char vert1_label = 25;
    const unsigned char vert2_label = 24;
    const unsigned char vert3_label = 23;
    const unsigned char vert4_label = 22;
    const unsigned char sacrum_label = 1;

    auto vert1_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert1_label, float(0), false);

    auto vert2_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert2_label, float(0), false);

    auto vert3_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert3_label, float(0), false);

    auto vert4_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert4_label, float(0), false);

    auto sacrum_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), sacrum_seg.GetPointer(), sacrum_label, float(0), false);

    auto vert1_rot_cen = ReadVertbRotCenter(vert_rot_fcsv, "vert1-rot");
    auto vert2_rot_cen = ReadVertbRotCenter(vert_rot_fcsv, "vert2-rot");
    auto vert3_rot_cen = ReadVertbRotCenter(vert_rot_fcsv, "vert3-rot");
    auto vert4_rot_cen = ReadVertbRotCenter(vert_rot_fcsv, "vert4-rot");

    ScalarType dist_v1_v2 = (vert1_rot_cen - vert2_rot_cen).norm();
    ScalarType dist_v2_v3 = (vert2_rot_cen - vert3_rot_cen).norm();
    ScalarType dist_v3_v4 = (vert3_rot_cen - vert4_rot_cen).norm();
    ScalarType rot_angX = 0.;
    ScalarType rot_angY = 0.;
    ScalarType rot_angZ = 0.;
    ScalarType transX = 0.;
    ScalarType transY = 0.;
    ScalarType transZ = 0.;
    //  Add
    AddFilterType::Pointer addFilter_vert = AddFilterType::New();

    // Resample Filters
    ResampleFilterType::Pointer resampleFilter_vert1 = ResampleFilterType::New();
    EulerTransformType::Pointer vert1_xform = EulerTransformType::New();
    rot_angX = _rot_angX_rand / 4.;
    rot_angY = _rot_angY_rand / 4.;
    rot_angZ = _rot_angZ_rand / 4.;
    TransformVertebrae(vert1_xform, rot_angX, rot_angY, rot_angZ, transX, transY, transZ, vert1_rot_cen);
    vout << "  [ResampleVertebrae] - Resampling vert1 ..." << std::endl
         << "                          rotX: " << rot_angX << std::endl
         << "                          rotY: " << rot_angY << std::endl
         << "                          rotZ: " << rot_angZ << std::endl
         << "                        transX: " << transX << std::endl
         << "                        transY: " << transY << std::endl
         << "                        transZ: " << transZ << std::endl
         << "                        Matrix:\n" << vert1_xform->GetMatrix() << std::endl;
    _vert1_xform4x4 = ComputeVertbAffineTransform(vert1_xform, transX, transY, transZ);
    vout << "                  AffineMatrix:\n" << _vert1_xform4x4.matrix() << std::endl;
    ResampleVolume(resampleFilter_vert1, vert1_vol, vert1_xform);
    auto vert1_resample = resampleFilter_vert1->GetOutput();

    ResampleFilterType::Pointer resampleFilter_vert2 = ResampleFilterType::New();
    EulerTransformType::Pointer vert2_xform = EulerTransformType::New();
    rot_angX += 2*_rot_angX_rand / 4.;
    rot_angY += 2*_rot_angY_rand / 4.;
    rot_angZ += 2*_rot_angZ_rand / 4.;
    transX += dist_v1_v2 * sin(rot_angY * kDEG2RAD) + dist_v1_v2  * (cos(rot_angZ * kDEG2RAD) - 1.);
    transY += -dist_v1_v2 * sin(rot_angX * kDEG2RAD) + dist_v1_v2 * sin(rot_angZ * kDEG2RAD);
    transZ += dist_v1_v2  * (cos(rot_angY * kDEG2RAD) - 1.) - dist_v1_v2  * (cos(rot_angX * kDEG2RAD) - 1.);
    TransformVertebrae(vert2_xform, rot_angX, rot_angY, rot_angZ, transX, transY, transZ, vert2_rot_cen);
    vout << "  [ResampleVertebrae] - Resampling vert2 ..." << std::endl
        << "                          rotX: " << rot_angX << std::endl
        << "                          rotY: " << rot_angY << std::endl
        << "                          rotZ: " << rot_angZ << std::endl
        << "                        transX: " << transX << std::endl
        << "                        transY: " << transY << std::endl
        << "                        transZ: " << transZ << std::endl
        << "                        Matrix:\n" << vert2_xform->GetMatrix() << std::endl;
    _vert2_xform4x4 = ComputeVertbAffineTransform(vert2_xform, transX, transY, transZ);
    vout << "                  AffineMatrix:\n" << _vert2_xform4x4.matrix() << std::endl;
    ResampleVolume(resampleFilter_vert2, vert2_vol, vert2_xform);
    auto vert2_resample = resampleFilter_vert2->GetOutput();

    ResampleFilterType::Pointer resampleFilter_vert3 = ResampleFilterType::New();
    EulerTransformType::Pointer vert3_xform = EulerTransformType::New();
    rot_angX += 3*_rot_angX_rand / 4.;
    rot_angY += 3*_rot_angY_rand / 4.;
    rot_angZ += 3*_rot_angZ_rand / 4.;
    transX += dist_v2_v3 * sin(rot_angY * kDEG2RAD) + dist_v2_v3  * (cos(rot_angZ * kDEG2RAD) - 1.);
    transY += -dist_v2_v3 * sin(rot_angX * kDEG2RAD) + dist_v2_v3 * sin(rot_angZ * kDEG2RAD);
    transZ += dist_v2_v3  * (cos(rot_angY * kDEG2RAD) - 1.) - dist_v2_v3  * (cos(rot_angX * kDEG2RAD) - 1.);
    TransformVertebrae(vert3_xform, rot_angX, rot_angY, rot_angZ, transX, transY, transZ, vert3_rot_cen);
    vout << "  [ResampleVertebrae] - Resampling vert3 ..." << std::endl
        << "                          rotX: " << rot_angX << std::endl
        << "                          rotY: " << rot_angY << std::endl
        << "                          rotZ: " << rot_angZ << std::endl
        << "                        transX: " << transX << std::endl
        << "                        transY: " << transY << std::endl
        << "                        transZ: " << transZ << std::endl
        << "                        Matrix:\n" << vert3_xform->GetMatrix() << std::endl;
    _vert3_xform4x4 = ComputeVertbAffineTransform(vert3_xform, transX, transY, transZ);
    vout << "                  AffineMatrix:\n" << _vert3_xform4x4.matrix() << std::endl;
    ResampleVolume(resampleFilter_vert3, vert3_vol, vert3_xform);
    auto vert3_resample = resampleFilter_vert3->GetOutput();

    ResampleFilterType::Pointer resampleFilter_vert4 = ResampleFilterType::New();
    EulerTransformType::Pointer vert4_xform = EulerTransformType::New();
    rot_angX += _rot_angX_rand;
    rot_angY += _rot_angY_rand;
    rot_angZ += _rot_angZ_rand;
    transX += dist_v3_v4 * sin(rot_angY * kDEG2RAD) + dist_v3_v4  * (cos(rot_angZ * kDEG2RAD) - 1.);
    transY += -dist_v3_v4 * sin(rot_angX * kDEG2RAD) + dist_v3_v4 * sin(rot_angZ * kDEG2RAD);
    transZ += dist_v3_v4  * (cos(rot_angY * kDEG2RAD) - 1.) - dist_v3_v4  * (cos(rot_angX * kDEG2RAD) - 1.);
    TransformVertebrae(vert4_xform, rot_angX, rot_angY, rot_angZ, transX, transY, transZ, vert4_rot_cen);
    vout << "  [ResampleVertebrae] - Resampling vert4 ..." << std::endl
        << "                          rotX: " << rot_angX << std::endl
        << "                          rotY: " << rot_angY << std::endl
        << "                          rotZ: " << rot_angZ << std::endl
        << "                        transX: " << transX << std::endl
        << "                        transY: " << transY << std::endl
        << "                        transZ: " << transZ << std::endl
        << "                        Matrix:\n" << vert4_xform->GetMatrix() << std::endl;
    _vert4_xform4x4 = ComputeVertbAffineTransform(vert4_xform, transX, transY, transZ);
    vout << "                  AffineMatrix:\n" << _vert4_xform4x4.matrix() << std::endl;
    ResampleVolume(resampleFilter_vert4, vert4_vol, vert4_xform);
    auto vert4_resample = resampleFilter_vert4->GetOutput();

    vout << "  [ResampleVertebrae] - Adding up vert2 ..." << std::endl;
    AddVolumes(addFilter_vert, vert1_resample, vert2_resample);
    auto TransImage_vert = addFilter_vert->GetOutput();

    vout << "  [ResampleVertebrae] - Adding up vert3 ..." << std::endl;
    AddVolumes(addFilter_vert, TransImage_vert, vert3_resample);
    TransImage_vert = addFilter_vert->GetOutput();

    vout << "  [ResampleVertebrae] - Adding up vert4 ..." << std::endl;
    AddVolumes(addFilter_vert, TransImage_vert, vert4_resample);
    TransImage_vert = addFilter_vert->GetOutput();

    vout << "  [ResampleVertebrae] - Adding up sacrum ..." << std::endl;
    AddVolumes(addFilter_vert, TransImage_vert, sacrum_vol);
    TransImage_vert = addFilter_vert->GetOutput();

    return TransImage_vert;
  }
}
