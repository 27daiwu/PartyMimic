# DEV PC only (never shipped): parity / FK checks and the loopback robot simulator for the exit-path checks.
add_executable(parity_test parity_test.cpp)
target_link_libraries(parity_test PRIVATE pm_policy)
add_executable(fk_test fk_test.cpp)
target_link_libraries(fk_test PRIVATE pm_policy)
add_executable(sim_robot sim_robot.cpp)
target_link_libraries(sim_robot PRIVATE pm_robot)
add_executable(gen_golden gen_golden.cpp)
target_link_libraries(gen_golden PRIVATE pm_policy)
