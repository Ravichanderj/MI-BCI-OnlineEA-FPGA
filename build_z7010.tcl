# Vivado 2019.2:  cd vivado ;  vivado -mode batch -source build_z7010.tcl
# Needs: hls/ synthesised + exported (run_hls.tcl).  NO board files needed:
#   on PYNQ the processor system (DDR, MIO, boot) is already configured by the PYNQ SD image;
#   Overlay() only loads the programmable logic and sets the PL clock from the .hwh.
#   So the PS here only needs the right part, GP0 (control), HP0 (data) and FCLK0 = 100 MHz.
# Output: mi_bci.bit + mi_bci.hwh  (copy both to the board, same folder, same base name)
set prj mi_bci_z10
create_project -force $prj ./$prj -part xc7z010clg400-1
set_property ip_repo_paths [list ../hls/mi_bci_prj/sol1/impl/ip] [current_project]
update_ip_catalog

create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} [get_bd_cells ps7]
# 100 MHz fabric clock, one high-performance port for the trial buffer in DDR
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells ps7]

create_bd_cell -type ip -vlnv sru.edu.in:bci:mi_bci_top:1.0 mi_bci_top_0
# Vivado HLS 2019.2 names the control port s_axi_AXILiteS
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins mi_bci_top_0/s_axi_AXILiteS]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/mi_bci_top_0/m_axi_gmem" Clk "Auto"} [get_bd_intf_pins ps7/S_AXI_HP0]
validate_bd_design
save_bd_design

make_wrapper -files [get_files ./$prj/$prj.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ./$prj/$prj.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v
set_property top design_1_wrapper [current_fileset]

# single-threaded: avoids the EXCEPTION_ACCESS_VIOLATION seen with parallel synthesis on Windows 11
set_param general.maxThreads 1
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1

file copy -force ./$prj/$prj.runs/impl_1/design_1_wrapper.bit ./mi_bci_z7010.bit
file copy -force ./$prj/$prj.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh ./mi_bci_z7010.hwh
open_run impl_1
report_utilization -file ./utilization_z7010.txt
report_timing_summary -file ./timing_z7010.txt
report_power -file ./power_estimate_z7010.txt
puts "DONE: vivado/mi_bci.bit + mi_bci.hwh, reports: utilization.txt timing.txt power_estimate.txt"
