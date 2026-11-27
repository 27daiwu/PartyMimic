# DEV PC only (never shipped): parity / FK checks and the loopback robot simulator for the exit-path checks.
add_executable(parity_test parity_test.cpp)
target_link_libraries(parity_test PRIVATE pm_policy)
add_executable(fk_test fk_test.cpp)
target_link_libraries(fk_test PRIVATE pm_policy)
add_executable(sim_robot sim_robot.cpp)
target_link_libraries(sim_robot PRIVATE pm_robot)
add_executable(gen_golden gen_golden.cpp)
target_link_libraries(gen_golden PRIVATE pm_policy)
add_executable(dds_stress dds_stress.cpp)
target_link_libraries(dds_stress PRIVATE pm_robot)
add_executable(log_stress log_stress.cpp)
target_link_libraries(log_stress PRIVATE pm_policy)
# V1.7 C3: isolated HARD_CAP / FALLEN checks of the StabilityMonitor + decide_exit + site.conf range checks (no robot, no DDS traffic). main.cpp is compiled with main renamed to get Site::load.
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/main.cpp PROPERTIES COMPILE_DEFINITIONS main=pm_real_main_unused)
add_executable(hard_cap_isolation hard_cap_isolation.cpp ${CMAKE_SOURCE_DIR}/src/main.cpp ${CMAKE_SOURCE_DIR}/src/runner.cpp ${CMAKE_SOURCE_DIR}/src/guardian.cpp ${CMAKE_SOURCE_DIR}/src/service.cpp
               ${CMAKE_SOURCE_DIR}/src/selfcheck.cpp ${CMAKE_SOURCE_DIR}/src/logtool.cpp)
target_link_libraries(hard_cap_isolation PRIVATE pm_robot)
