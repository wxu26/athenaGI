For every new feature I added, run the following tests:

0. blast.in (sanity check that my new physcis does not cause problem when not used)

1. test_remap.in (test RemapColumns; check output message for test result)

2. test_sph_gravity.in (test SphGravity on a non-uniform grid; check output message for error; also see plots in test_sph_gravity.ipynb)

3. test_optical_depth.in (test OpticalDepth in standalone physics; check result using test_optical_dpeth.ipynb)