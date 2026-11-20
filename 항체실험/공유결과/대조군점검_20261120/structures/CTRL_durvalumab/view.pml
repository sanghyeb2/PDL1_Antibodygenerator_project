reinitialize
load native_Fv.pdb, native_Fv
load predicted_Fv_aligned.pdb, predicted_Fv
load native_antigen.pdb, native_antigen
load predicted_antigen_aligned.pdb, predicted_antigen
hide everything
show cartoon
color gray70, native_Fv or native_antigen
color teal, predicted_Fv
color orange, predicted_antigen
disable all
enable native_antigen
enable predicted_antigen
orient native_antigen
scene antigen_independent_fit, store
disable all
enable native_Fv
enable predicted_Fv
orient native_Fv
scene Fv_VH_framework_fit, store
