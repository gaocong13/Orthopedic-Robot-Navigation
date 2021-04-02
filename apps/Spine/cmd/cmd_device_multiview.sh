RootPath=/home/cong/Research/Spine/CadaverNeedleInjection/Registration/Mar16
CurPath=/home/cong/Research/Spine/CadaverNeedleInjection
LandmarkAnnot=${RootPath}/landmark_annotation
MetaData=${CurPath}/meta_data
ImageDCM=${RootPath}/RealXraydcm
Initxform=${CurPath}/output_singleview
Output=${CurPath}/output_multiview
ExpTXT=${CurPath}/expID.txt
xreg-spine-device-regi-multiview ${LandmarkAnnot} ${MetaData} ${ImageDCM} ${Initxform} ${Output} ${ExpTXT} -v
