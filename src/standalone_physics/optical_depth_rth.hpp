#ifndef STANDALONE_OPTICAL_DEPTH_RTH_HPP_
#define STANDALONE_OPTICAL_DEPTH_RTH_HPP_

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../fft/remap_columns.hpp"

class MeshBlock;


//----------------------------------------------------------------------------------------
//! \class OpticalDepth
//! \brief estimate optical depth with pseudo ray tracing

// rays are defined in r-th direction:
// start from a given cell, first move outward/inward in r and then move upward/downward in th until reaching a pole.

// other forms of rays are possible; e.g., th-r or phi-r-th

class OpticalDepth: public RemapColumns {
  public:
    AthenaArray<Real> rho_kappa; // rho * kappa; needs to be set externally
    AthenaArray<Real> weights_and_tau;
      // cooling weights, diffusion weights, and tau cc
      // these are kept in the same array for easier boundary communication
    bool keep_tau_cc = false; // not storing tau_cc slightly reduces cost
      // average of exp(-tau) in the current cell
    void Initialize(MeshBlock * my_pmb);
    void EstimateOpticalDepth();
  private:
    bool reflect_;
    Real *s0, *s1, *s2, *s3, *s4, *s5;
    const int COOL=0, DIFF=1, TAU=2;
    void GetTauTop(Real * rho_kappa, Real * tau_top);
    void GetTauCCAndWeights(Real * rho_kappa, Real * tau_top, Real * tau_cc, Real * weight_cooling, Real * weight_diffusion);
};


#endif // STANDALONE_OPTICAL_DEPTH_RTH_HPP_