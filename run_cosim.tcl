# Vivado HLS 2019.2:  cd hls ;  vivado_hls -f run_cosim.tcl      (about 1-2 hours in total)
# Three exact C/RTL co-simulations on 2 trials of S1:  EA half, CNN half, full top.
# Run as:  vivado_hls -f run_cosim.tcl > cosim_all.log   then send cosim_all.log
foreach {mode top} {EA ea_wrap CNN cnn_wrap TOP mi_bci_top} {
    open_project -reset cosim_$mode
    set_top $top
    add_files mi_bci_top.cpp -cflags "-std=c++0x"
    add_files online_ea.cpp  -cflags "-std=c++0x"
    add_files hanet_cnn.cpp  -cflags "-std=c++0x"
    add_files cosim_wrap.cpp -cflags "-std=c++0x"
    add_files -tb tb_cosim.cpp -cflags "-std=c++0x -DMODE_$mode"
    add_files -tb cosim_in.bin
    add_files -tb cosim_q.bin
    add_files -tb cosim_log.bin
    open_solution -reset sol1
    set_part {xc7z020clg400-1}
    create_clock -period 10 -name default
    csim_design
    csynth_design
    if {[catch {cosim_design -rtl verilog -tool xsim} err]} { puts "COSIM $mode returned: $err" }
    close_project
}
exit
