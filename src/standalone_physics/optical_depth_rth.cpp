#include <iostream>
#include <sstream>
#include <algorithm>    // std::min,max

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../fft/remap_columns.hpp"
#include "../utils/buffer_utils.hpp"

#include "optical_depth_rth.hpp"

#ifdef MPI_PARALLEL // the remaps used in this class requires MPI

void OpticalDepth::Initialize(MeshBlock * my_pmb) {
  RemapColumns::Initialize(my_pmb);
  rho_kappa.NewAthenaArray(pmb->ncells3,pmb->ncells2,pmb->ncells1);
  int n = keep_tau_cc ? 3 : 2;
  weights_and_tau.NewAthenaArray(n,pmb->ncells3,pmb->ncells2,pmb->ncells1);
  InitializeBoundary(&weights_and_tau);
  // check the bounds
  Real small_number = 1.e-8;
  int Nj = N[1];
  if (std::abs(xf[1](0))>small_number or
      (std::abs(xf[1](Nj)-0.5*PI)>small_number and
       std::abs(xf[1](Nj)-PI)>small_number)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SphGravity" << std::endl
        << "Only theta=[0,pi] or [0,pi/2] is supported"<< std::endl;
    ATHENA_ERROR(msg);
  }
  reflect_ = std::abs(xf[1](Nj)-0.5*PI)<=small_number;
  s0 = new Real[cnt];
  s1 = new Real[cnt];
  s2 = new Real[cnt];
  s3 = new Real[cnt];
  s4 = new Real[cnt];
  s5 = new Real[cnt];
}

void OpticalDepth::EstimateOpticalDepth() {
  // assume that kappa has been set elsewhere
  // initialize buffers
  Real * rho_kappa = s0;
  Real * tau_top = s1;
  Real * tau_cc = s2;
  Real * weight_cooling = s3;
  Real * weight_diffusion = s4;
  Real * scratch = s5;
  // load data
  int offset_in = 0;
  BufferUtility::PackData(this->rho_kappa, rho_kappa,
    pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke, offset_in);
  // remap to theta
  std::swap(scratch, rho_kappa);
  Remap(0,2,scratch, rho_kappa);
  // optical depth along theta
  GetTauTop(rho_kappa, tau_top);
  // remap to r
  std::swap(scratch, rho_kappa);
  Remap(2,1,scratch, rho_kappa);
  std::swap(scratch, tau_top);
  Remap(2,1,scratch, tau_top);
  // tau cc and weights
  GetTauCCAndWeights(rho_kappa, tau_top, tau_cc, weight_cooling, weight_diffusion);
  // remap to meshblock
  std::swap(scratch, weight_cooling);
  Remap(1,0,scratch, weight_cooling);
  std::swap(scratch, weight_diffusion);
  Remap(1,0,scratch, weight_diffusion);
  if (keep_tau_cc) {
    std::swap(scratch, tau_cc);
    Remap(1,0,scratch, tau_cc);
  }
  // save data
  int offset_out = 0;
  BufferUtility::UnpackData(weight_cooling, weights_and_tau,
    COOL, COOL, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke, offset_out);
  offset_out = 0;
  BufferUtility::UnpackData(weight_diffusion, weights_and_tau,
    DIFF, DIFF, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke, offset_out);
  if (keep_tau_cc) {
    offset_out = 0;
    BufferUtility::UnpackData(tau_cc, weights_and_tau,
      TAU, TAU, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke, offset_out);
  }
  // apply bc
  SetBoundaries(); // this uses outflow bc when a user r bc is specified
}


void OpticalDepth::GetTauTop(Real * rho_kappa, Real * tau_top) {
  const int ni=n[2][0], nj=n[2][1], nk=n[2][2];
  const int isi=is[2][0];
  for (int i=0; i<ni; ++i) {
    Real r = x[0](isi+i);
    for (int k=0; k<nk; ++k) {
      int ind = i*nk*nj + k*nj;
      // first sweep: increasing th
      tau_top[ind] = 0.;
      for (int j=0; j<nj-1; ++j)
        tau_top[ind+j+1] = tau_top[ind+j] + (xf[1](j+1)-xf[1](j))*r*rho_kappa[ind+j];
      // second sweep: decreasing th
      if (reflect_) continue;
      tau_top[ind+nj-1] = 0.;
      for (int j=nj-1; j>0; --j)
        tau_top[ind+j-1] = std::min(tau_top[ind+j-1],
          tau_top[ind+j] + (xf[1](j+1)-xf[1](j))*r*rho_kappa[ind+j]);
    }
  }
  return;
}

Real AvgExpMinusTau(Real dtau) {
  const Real small_number = 1.e-8;
  dtau += small_number;
  return (1.-std::exp(-dtau))/dtau;
}

void OpticalDepth::GetTauCCAndWeights(Real * rho_kappa, Real * tau_top, Real * tau_cc, Real * weight_cooling, Real * weight_diffusion) {

  // for cooling weight exp(-tau), we average along the ray because using cc tau can introduce large error
  // when the ray goes from tau->0 to some large tau within a cell
  
  // for diffusion weight exp(-1/tau), we use cc tau because that does not produce much error and a more
  // proper averaging is expensive

  const int ni=n[1][0], nj=n[1][1], nk=n[1][2];
  const int isj=is[1][1];
  for (int k=0; k<nk; ++k) {
    for (int j=0; j<nj; ++j) {
      int ind = k*nj*ni + j*ni;
      // initialize: along theta
      for (int i=0; i<ni; ++i) {
        Real dtau_th = rho_kappa[ind+i]*x[0](i)*(xf[1](isj+j+1)-xf[1](isj+j));
        tau_cc[ind+i] = tau_top[ind+i] + .5*dtau_th;
        weight_cooling[ind+i] = std::exp(-tau_top[ind+i]) * AvgExpMinusTau(dtau_th);
      }
      // sweep inside out
      Real tau_left = 0.;
      for (int i=0; i<ni; ++i) {
        Real dtau_r = rho_kappa[ind+i]*(xf[0](i+1)-xf[0](i));
        tau_cc[ind+i] = std::min(tau_cc[ind+i], tau_left+.5*dtau_r);
        Real weight_cooling_r = std::exp(-tau_left) * AvgExpMinusTau(dtau_r);
        weight_cooling[ind+i] = std::max(weight_cooling[ind+i], weight_cooling_r);
        tau_left = tau_cc[ind+i] + .5*dtau_r;
      }
      // sweep outside in
      tau_left = 0.;
      for (int i=ni-1; i>=0; --i) {
        Real dtau_r = rho_kappa[ind+i]*(xf[0](i+1)-xf[0](i));
        tau_cc[ind+i] = std::min(tau_cc[ind+i], tau_left+.5*dtau_r);
        Real weight_cooling_r = std::exp(-tau_left) * AvgExpMinusTau(dtau_r);
        weight_cooling[ind+i] = std::max(weight_cooling[ind+i], weight_cooling_r);
        tau_left = tau_cc[ind+i] + .5*dtau_r;
      }
      // update diffusion weight
      for (int i=0; i<ni; ++i)
        weight_diffusion[ind+i] = std::exp(-1./(tau_cc[ind+i]+TINY_NUMBER));
    }
  }
  return;
}

#endif