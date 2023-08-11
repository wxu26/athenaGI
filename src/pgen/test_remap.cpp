#include <iostream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#include "../fft/remap_columns.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif


void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // do nothing here
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  
  MeshBlock *pmb = my_blocks(0);

  // initialize a remap column object
  RemapColumns R;
  R.Initialize(pmb);

  R.Test();
  // alternatively, use
  // R.Test(int layout_in=0123, int layout_out=0123, int axis=012, int print=012)
  // to test individual layout and axis
  // print=012 specify different amount of output

}