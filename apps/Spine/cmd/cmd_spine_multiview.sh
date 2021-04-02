CurPath=/home/cong/Research/Spine/CadaverNeedleInjection
RootPath=/home/cong/Research/Spine/CadaverNeedleInjection/Registration/Mar16
MetaData=${CurPath}/meta_data
ImageDCM=${RootPath}/RealXraydcm
InitFolder=${CurPath}/output_device_singleview
Output=${CurPath}/output_spine_multiview
DeviceExpTXT=${CurPath}/device_expID.txt
SpineExpTXT=${CurPath}/spine_expID.txt

Output=${CurPath}/output_spine_singleview
xreg-spine-spine-regi-multiview ${MetaData} ${ImageDCM} ${InitFolder} ${MetaData} ${ExpTXT}  ${Output} -v
