# Vivado HLS 2019.2:  cd hls ;  vivado_hls -f run_hls.tcl
# 1) C simulation (end-to-end test bench)  2) synthesis  3) export IP for Vivado
open_project -reset mi_bci_prj
set_top mi_bci_top
add_files mi_bci_top.cpp -cflags "-std=c++0x"
add_files online_ea.cpp  -cflags "-std=c++0x"
add_files hanet_cnn.cpp  -cflags "-std=c++0x"
add_files -tb tb_mi_bci_top.cpp -cflags "-std=c++0x"
add_files -tb ea_in.bin
open_solution -reset sol1
# PYNQ-Z2. For the EDGE Z7010 board use: set_part {xc7z010clg400-1}
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vendor sru.edu.in -library bci -version 1.0 -display_name "MI-BCI online-EA + int8 CNN"
csim_design
csynth_design
export_design -format ip_catalog
exit
