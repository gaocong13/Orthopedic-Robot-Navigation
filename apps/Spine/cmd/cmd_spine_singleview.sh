CurPath=/home/cong/Research/Spine/CadaverNeedleInjection
RootPath=/home/cong/Research/Spine/CadaverNeedleInjection/Registration/Mar16
LandmarkAnnot=${RootPath}/landmark_annotation
MetaData=${CurPath}/meta_data
ExpTXT=${CurPath}/expID.txt
ImageDCM=${RootPath}/RealXraydcm
Output=${CurPath}/output_spine_singleview
xreg-spine-spine-regi-singleview ${LandmarkAnnot} ${MetaData} ${ExpTXT} ${ImageDCM} ${Output} -v
