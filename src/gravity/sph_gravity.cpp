//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sph_gravity.cpp
//! \brief implementation of functions in class SphGravity

#define SPH_GRAV_DEBUG 0

// C headers

// C++ headers
#include <iostream>  // cout
#include <sstream>    // sstream

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../fft/remap_columns.hpp"
//#include "../fft/plimpton/remap_3d.h"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../task_list/grav_task_list.hpp"
#include "../utils/buffer_utils.hpp"
#include "sph_gravity.hpp"
#include "gravity.hpp"
#ifdef FFT
#include "fftw3.h"
#if SELF_GRAVITY_ENABLED == 3
#include <Eigen/Dense>

#include <vector>
#include <numeric>      // iota
#include <algorithm>    // stable_sort

//----------------------------------------------------------------------------------------
//! \fn std::vector<int> argsort(const std::vector<Real> &v)
//! \brief argsort function

std::vector<int> argsort(const std::vector<Real> &v) {
  std::vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2];});
  return idx;
}

//----------------------------------------------------------------------------------------
//! \fn SphGravityDriver::SphGravityDriver(Mesh *pm, ParameterInput *pin)
//! \brief SphGravityDriver constructor

SphGravityDriver::SphGravityDriver(Mesh *pm, ParameterInput *pin) {
  pmy_mesh_ = pm;
  four_pi_G_ = pmy_mesh_->four_pi_G_;
  if (four_pi_G_ == 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in GravityDriver::GravityDriver" << std::endl
        << "Gravitational constant must be set in the Mesh::InitUserMeshData "
        << "using the SetGravitationalConstant or SetFourPiG function." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }
  one_solve_per_cycle_ = pin->GetOrAddBoolean(
    "gravity","one_solve_per_cycle",one_solve_per_cycle_);
  grav_.M_star = pin->GetOrAddReal("gravity","M_star",grav_.M_star);
  grav_.x_star = pin->GetOrAddReal("gravity","x_star",grav_.x_star);
  grav_.y_star = pin->GetOrAddReal("gravity","y_star",grav_.y_star);
  grav_.z_star = pin->GetOrAddReal("gravity","z_star",grav_.z_star);
  grav_.r_smooth = pin->GetOrAddReal("gravity","r_smooth",grav_.r_smooth);
  //gtlist_ = new GravityBoundaryTaskList(pin, pm);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn SphGravityDriver::~SphGravityDriver()
//! \brief SphGravityDriver destructor

SphGravityDriver::~SphGravityDriver() {
  //delete gtlist_;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravityDriver::Solve(int stage, int mode)
//! \brief solve for gravitational potential

void SphGravityDriver::Solve(int stage) {
  if (SPH_GRAV_DEBUG) std::cout<<"start solve"<<std::endl;
  // check if we want to skip this solve
  if (one_solve_per_cycle_ && grav_.Initialized() && stage!=1) return;
  // initialize if necessary
  if (!grav_.Initialized()) {
    grav_.Initialize(pmy_mesh_->my_blocks(0), four_pi_G_);
  }
  // solve self gravity
  grav_.Solve();
  // apply bcs
  grav_.SetBoundaries();
  // add point mass (include in ghost cells)
  if (use_update_star_fn_) update_star_fn_(&grav_, pmy_mesh_->my_blocks(0));
  grav_.AddPointMass();
  if (SPH_GRAV_DEBUG) std::cout<<"finish solve"<<std::endl;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::Initialize(int stage, int mode)
//! \brief initialize SphGravity

void SphGravity::Initialize(MeshBlock * my_pmb, Real four_pi_G){
  RemapColumns::Initialize(my_pmb);
  this->four_pi_G = four_pi_G;
  InitializePhi();
  InitializeTh();
  InitializeR();
  InitializeBoundary(&pmb->pgrav->phi);
  s0 = new Real[cnt];
  s1 = new Real[cnt];
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::Solve()
//! \brief solve for self gravity

void SphGravity::Solve(){
  Real * data = s0;
  Real * scratch = s1;
  // load source
  LoadSource(data);
  // remap to phi
  std::swap(data, scratch);
  Remap(0, 3, scratch, data);
  // process phi (fft)
  RFFT(data, 1);
  // remap to theta
  std::swap(data, scratch);
  Remap(3, 2, scratch, data);
  // process theta (transform to sherical harmonics basis)
  std::swap(data, scratch);
  GridToBasis(scratch, data, 1);
  // remap to r
  std::swap(data, scratch);
  Remap(2, 1, scratch, data);
  // process r
  std::swap(data, scratch);
  SolveR(scratch, data);
  // remap to theta
  std::swap(data, scratch);
  Remap(1, 2, scratch, data);
  // process theta
  std::swap(data, scratch);
  GridToBasis(scratch, data, -1);
  // remap to phi
  std::swap(data, scratch);
  Remap(2, 3, scratch, data);
  // process phi
  RFFT(data, -1);
  // remap back to original layout
  std::swap(data, scratch);
  Remap(3, 0, scratch, data);
  // store phi and apply bc (now we only do radial physical bc)
  StorePhi(data);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::SetBoundaries()
//! \brief set phi in ghost zones

void SphGravity::SetBoundaries(){
  RemapColumns::SetBoundaries();
  RadialBoundaries();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::AddPointMass()
//! \brief add point mass

void SphGravity::AddPointMass(){
  if (M_star==0) return;
  Real GM = M_star * four_pi_G / (4.*PI);
  Real r_smooth_3 = r_smooth*r_smooth*r_smooth;
  int il=pmb->is-NGHOST, iu=pmb->ie+NGHOST;
  int jl=pmb->js, ju=pmb->je, kl=pmb->ks, ku=pmb->ke;
  if (N[1]>1) {jl-=NGHOST; ju+=NGHOST;}
  if (N[2]>1) {kl-=NGHOST; ku+=NGHOST;}
  
  if (x_star==0 && y_star==0 && z_star==0) { // point mass at origin
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real d = pmb->pcoord->x1v(i);
          Real f = (d>r_smooth) ? 
            -1./d : -1./r_smooth + .5*(SQR(d)-SQR(r_smooth))/r_smooth_3;
          pmb->pgrav->phi(k,j,i) += GM*f;
        }
      }
    }
  } else { // point mass not at origin
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real x0=1., y0=1., z0=1.;
          Real x1, y1, z1;
          Real r, th, phi;
          r = pmb->pcoord->x1v(i);
          th = pmb->pcoord->x2v(j);
          phi = pmb->pcoord->x3v(k);
          x1 = r*std::sin(th)*std::cos(phi);
          y1 = r*std::sin(th)*std::sin(phi);
          z1 = r*std::cos(th);
          Real d = std::sqrt(SQR(x1-x_star) + SQR(y1-y_star) + SQR(z1-z_star));
          Real f = (d>r_smooth) ? 
            -1./d : -1./r_smooth + .5*(SQR(d)-SQR(r_smooth))/r_smooth_3;
          pmb->pgrav->phi(k,j,i) += GM*f;
        }
      }
    }
  }  
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::LoadSource()
//! \brief load the (scaled) source term of the Poisson equation

void SphGravity::LoadSource(Real * src) {
  // load 4 pi G rho r^2 / Nphi
  // the 1/Nphi term is for fft normalization
  for (int k=pmb->ks;k<=pmb->ke;k++) {
    for (int j=pmb->js;j<=pmb->je;j++) {
      for (int i=pmb->is;i<=pmb->ie;i++) {
        int ind = (k-pmb->ks)*pmb->block_size.nx2*pmb->block_size.nx1 
                + (j-pmb->js)*pmb->block_size.nx1 + (i-pmb->is);
        src[ind] = pmb->phydro->u(IDN,k,j,i) * four_pi_G * SQR(pmb->pcoord->x1v(i)) / N[2];
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::RFFT(Real * data, int dir)
//! \brief rfft in phi direction

void SphGravity::RFFT(Real * data, int dir) {
  // not 3d: no need to do anything
  if (N[2]==1) return;
  if (dir== 1) fftw_execute_r2r(r2r_fft_fwd,data,data);
  if (dir==-1) fftw_execute_r2r(r2r_fft_bck,data,data);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::GridToBasis(Real * data_in, Real * data_out, int dir)
//! \brief transform to/from discrete spherical harmonics in theta direction

void SphGravity::GridToBasis(Real * data_in, Real * data_out, int dir) {
  int ni=n[2][0], nj=n[2][1], nk=n[2][2];
  if (dir==1) {
    for (int i=0;i<ni;i++) {
      for (int k=0;k<nk;k++) {
        int ind0 = k*nj + i*nj*nk;
        for (int j=0;j<nj;j++) {
          data_out[ind0+j] = 0.;
          for (int j2=0;j2<nj;j2++) {
            data_out[ind0+j] += data_in[ind0+j2]*grid_to_basis(k,j,j2);
          }
        }
      }
    }
  }
  else if (dir==-1) {
    for (int i=0;i<ni;i++) {
      for (int k=0;k<nk;k++) {
        int ind0 = k*nj + i*nj*nk;
        for (int j=0;j<nj;j++) {
          data_out[ind0+j] = 0.;
          for (int j2=0;j2<nj;j2++) {
            data_out[ind0+j] += data_in[ind0+j2]*basis_to_grid(k,j,j2);
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::SolveR(Real * src, Real * phi)
//! \brief solve the radial Poisson equation in discrete spherical harmonics basis

void SphGravity::SolveR(Real * src, Real * phi) {
  // solve using precomputed LU decomposition
  int nr=N[0], ni=n[1][0], nj=n[1][1], nk=n[1][2];
  Real * x = new Real[nr];
  Real * y = new Real[nr];
  Real * b = new Real[nr];
  int iAm2=0, iAm1=1, iA0=2, iA1=3, iA2=4, iL1=1, iL2=0, iU0=2, iU1=3, iU2=4;
  int is1=is[1][0], js1=is[1][1], ks1=is[1][2];
  for (int k=0; k<nk; k++) {
    for (int j=0; j<nj; j++) {
      int im = ks1+k;
      int il = js1+j;
      int ind0 = j*ni + k*nj*ni;
      for (int i=0; i<nr; i++) b[i] = src[i+ind0];
      y[0]=b[0];
      y[1]=b[1]-LU(k,j,iL1,0)*y[0];
      for (int i=0; i<nr-2; i++)
        y[i+2] = b[i+2]-LU(k,j,iL1,i+1)*y[i+1]-LU(k,j,iL2,i)*y[i];
      x[nr-1] = y[nr-1]/LU(k,j,iU0,nr-1);
      x[nr-2] = (y[nr-2]-LU(k,j,iU1,nr-2)*x[nr-1])/LU(k,j,iU0,nr-2);
      for (int i=nr-3; i>=0; i--)
        x[i] = (y[i]-LU(k,j,iU1,i)*x[i+1]-LU(k,j,iU2,i)*x[i+2])/LU(k,j,iU0,i);
      for (int i=0; i<nr; i++) phi[i+ind0] = x[i];
    }
  }
  delete[] x;
  delete[] y;
  delete[] b;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::StorePhi(Real * phi)
//! \brief store result to pmb->pgrav->phi

void SphGravity::StorePhi(Real * phi) {
  int offset = 0;
  BufferUtility::UnpackData(phi, pmb->pgrav->phi,
    pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke, offset);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::RadialBoundaries()
//! \brief radial boundary conditions

void SphGravity::RadialBoundaries() {
  int jl=pmb->js, ju=pmb->je, kl=pmb->ks, ku=pmb->ke;
  if (N[1]>1) {jl-=NGHOST; ju+=NGHOST;}
  if (N[2]>1) {kl-=NGHOST; ku+=NGHOST;}
  // inner boundary
  if (is[0][0]==0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        Real phi0 = pmb->pgrav->phi(k,j,pmb->is);
        Real dphi = pmb->pgrav->phi(k,j,pmb->is+1)-pmb->pgrav->phi(k,j,pmb->is);
        for (int i=1; i<=NGHOST; ++i) {
          pmb->pgrav->phi(k,j,pmb->is-i) = phi0 - dphi*i;
        }
      }
    }
  }
  // outer boundary
  if (ie[0][0]==N[0]-1) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        Real phi0 = pmb->pgrav->phi(k,j,pmb->ie);
        Real dphi = pmb->pgrav->phi(k,j,pmb->ie-1)-pmb->pgrav->phi(k,j,pmb->ie);
        for (int i=1; i<=NGHOST; ++i) {
          pmb->pgrav->phi(k,j,pmb->ie+i) = phi0 - dphi*i;
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void SphGravity::InitializeR()
//! \brief precompute LU for the radial equation

void SphGravity::InitializeR(){
  // check if r grid is log uniform
  Real r_min = pmb->pmy_mesh->mesh_size.x1min;
  Real r_max = pmb->pmy_mesh->mesh_size.x1max;
  Real r_ratio = pmb->pmy_mesh->mesh_size.x1rat;
  Real r_ratio_exact = std::pow(r_max/r_min, 1./N[0]);
  if (std::abs(r_ratio-r_ratio_exact)>1.e-8) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SphGravity" << std::endl
        << "Only log-uniform radial grid is supported"<< std::endl;
    ATHENA_ERROR(msg);
  }
  dlnr = std::log(r_ratio);
  // generate LU matrices
  // coefficients for the r component of the Laplace operator
  // (4th order finite difference)
  Real cll, cl, cc0, cr, crr;
  cll =  1./12./dlnr - 1./12./dlnr/dlnr;
  cl  =  -2./3./dlnr + 4./3./dlnr/dlnr;
  cc0 =  -5./2./dlnr/dlnr;
  cr  =   2./3./dlnr + 4./3./dlnr/dlnr;
  crr =  -1./12./dlnr - 1./12./dlnr/dlnr;
  // store both the Poisson operator A and the LU matrix in banded form:
  // each diagnoal (or offseted diagnoal) is a row
  // example: 3 diagnoals of a 3x3 matrix
  // [x,x,0] // offset by 1
  // [x,x,x] // diagnoal
  // [x,x,0] // offset by -1
  AthenaArray <Real> A;
  int ni1=n[1][0], nj1=n[1][1], nk1=n[1][2];
  LU.NewAthenaArray(nk1,nj1,5,ni1);
  A.NewAthenaArray(nk1,nj1,5,ni1);
  int iAm2=0, iAm1=1, iA0=2, iA1=3, iA2=4, iL1=1, iL2=0, iU0=2, iU1=3, iU2=4;
  int is1=is[1][0], js1=is[1][1], ks1=is[1][2];
  for (int k=0; k<nk1; k++) {
    for (int j=0; j<nj1; j++) {
      int im = ks1+k;
      int il = js1+j;
      Real dsq_thphi_eigen_current = dsq_thphi_eigen(im,il);
      Real l = .5*(-1.+std::sqrt(1.-4.*dsq_thphi_eigen_current));
        // solve -l(l+1) = dsq_thphi_eigen_current
        // l^2 + l + dsq_thphi_eigen_current = 0
      Real xp = std::pow(r_ratio, l), xm=std::pow(r_ratio, -(l+1));
        // phi(i+-1)/phi(i) beyond inner/outer boundary
      Real cc = cc0 + dsq_thphi_eigen_current;
      // initializa A
      for (int i=0; i<ni1-2; i++) A(k,j,iAm2,i)=cll;
      for (int i=0; i<ni1-1; i++) A(k,j,iAm1,i)=cl;
      for (int i=0; i<ni1  ; i++) A(k,j,iA0 ,i)=cc;
      for (int i=0; i<ni1-1; i++) A(k,j,iA1 ,i)=cr;
      for (int i=0; i<ni1-2; i++) A(k,j,iA2 ,i)=crr;
      A(k,j,iA0 ,0) += cll/xp/xp + cl/xp;
      A(k,j,iAm1,0) += cll/xp;
      A(k,j,iA0 ,ni1-1) += cr*xm + crr*xm*xm;
      A(k,j,iA1 ,ni1-2) += crr*xm;
      // U2
      for (int i=0; i<ni1-2; i++) LU(k,j,iU2 ,i)=A(k,j,iA2 ,i);
      // first iteration
      LU(k,j,iU0,0) = A(k,j,iA0,0);
      // second iteration
      LU(k,j,iU1,0) = A(k,j,iA1,0);
      LU(k,j,iL1,0) = A(k,j,iAm1,0)/LU(k,j,iU0,0);
      LU(k,j,iU0,1) = A(k,j,iA0,1)-LU(k,j,iL1,0)*LU(k,j,iU1,0);
      // other iterations
      for (int i=0; i<ni1-2; i++) {
        LU(k,j,iL2,i) = A(k,j,iAm2,i)/LU(k,j,iU0,i);
        LU(k,j,iU1,i+1) = A(k,j,iA1,i+1)-LU(k,j,iL1,i)*LU(k,j,iU2,i);
        LU(k,j,iL1,i+1) = (A(k,j,iAm1,i+1)-LU(k,j,iL2,i)*LU(k,j,iU1,i))/LU(k,j,iU0,i+1);
        LU(k,j,iU0,i+2) = A(k,j,iA0,i+2) - LU(k,j,iL2,i)*LU(k,j,iU2,i) - LU(k,j,iL1,i+1)*LU(k,j,iU1,i+1);
      }
    }
  }
  A.DeleteAthenaArray();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SphGravity::InitializeTh()
//! \brief precompute transformation to/from discrete Legendre polynomials
//!        and compute theta-phi eigenvalues

void SphGravity::InitializeTh(){
  int Nj=N[1], Nk=N[2], nk2=n[2][2];
  int nth = Nj;
  AthenaArray<Real> & x2f = xf[1];
  AthenaArray<Real> & x2 = x[1];
  // domain has to be 0~pi or 0~pi/2
  Real small_number = 1.e-8;
  if (std::abs(x2f(0))>small_number or
      (std::abs(x2f(Nj)-0.5*PI)>small_number and
       std::abs(x2f(Nj)-PI)>small_number)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in SphGravity" << std::endl
        << "Only theta=[0,pi] or [0,pi/2] is supported"<< std::endl;
    ATHENA_ERROR(msg);
  }
  // whether the outer theta bc is reflecting
  bool reflect = std::abs(x2f(Nj)-0.5*PI)<small_number;
  //std::cout<<"reflect = "<<reflect<<std::endl;
  // initialization
  grid_to_basis.NewAthenaArray(nk2,Nj,Nj);
  basis_to_grid.NewAthenaArray(nk2,Nj,Nj);
  dsq_thphi_eigen.NewAthenaArray(Nk,Nj);
  // step 1. make derivative matrices
  Eigen::MatrixXd Dth_even = Eigen::MatrixXd::Zero(nth,nth);
  Eigen::MatrixXd DDth_even = Eigen::MatrixXd::Zero(nth,nth);
  Eigen::MatrixXd Dth_odd = Eigen::MatrixXd::Zero(nth,nth);
  Eigen::MatrixXd DDth_odd = Eigen::MatrixXd::Zero(nth,nth);
  Eigen::MatrixXd Dth = Eigen::MatrixXd::Zero(nth,nth);
  Eigen::MatrixXd DDth = Eigen::MatrixXd::Zero(nth,nth);
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nth,nth);
  Eigen::MatrixXd s(5,5); // scratch
  Eigen::VectorXd b(5); // scratch
  Eigen::VectorXd w(5); // scratch
  // extend theta
  Real * thlong = new Real[nth+4];
  for (int j=0;j<nth;++j) {
    thlong[j+2] = x2(j);
  }
  thlong[0] = -x2(1);
  thlong[1] = -x2(0);
  thlong[nth+2] = 2*x2f(nth)-x2(nth-1);
  thlong[nth+3] = 2*x2f(nth)-x2(nth-2);
  // make matrices
  for (int j=0; j<nth; j++) {
    int jll=j-2, jl=j-1, jr=j+1, jrr=j+2;
    for (int i=0; i<=4; i++){
      s(i,0) = std::pow(thlong[j  ]-thlong[j+2],i);
      s(i,1) = std::pow(thlong[j+1]-thlong[j+2],i);
      s(i,2) = std::pow(0.,i);
      s(i,3) = std::pow(thlong[j+3]-thlong[j+2],i);
      s(i,4) = std::pow(thlong[j+4]-thlong[j+2],i);
    }
    // weights for first derivative
    b << 0., 1., 0., 0., 0.;
    w = s.colPivHouseholderQr().solve(b);
    Real wlle=w[0], wle=w[1], wce=w[2], wre=w[3], wrre=w[4];
    Real wllo=w[0], wlo=w[1], wco=w[2], wro=w[3], wrro=w[4];
    // weights for second derivative
    b << 0., 0., 2., 0., 0.;
    w = s.colPivHouseholderQr().solve(b);
    Real w2lle=w[0], w2le=w[1], w2ce=w[2], w2re=w[3], w2rre=w[4];
    Real w2llo=w[0], w2lo=w[1], w2co=w[2], w2ro=w[3], w2rro=w[4];
    // adjust for boundary
    if (jll<0) {
      jll=-1-jll;
      wllo*=-1;
      w2llo*=-1;
    }
    if (jl<0) {
      jl=-1-jl;
      wlo*=-1;
      w2lo*=-1;
    }
    if (jr>=nth) {
      jr=2*nth-1-jr;
      if (!reflect) { // only enforce odd parity when the outer bdry is also a pole
        wro*=-1;
        w2ro*=-1;
      }
    }
    if (jrr>=nth) {
      jrr=2*nth-1-jrr;
      if (!reflect) {
        wrro*=-1;
        w2rro*=-1;
      }
    }
    Dth_even(j,jll) += wlle;
    Dth_even(j,jl)  += wle;
    Dth_even(j,j)   += wce;
    Dth_even(j,jr)  += wre;
    Dth_even(j,jrr) += wrre;
    Dth_odd(j,jll) += wllo;
    Dth_odd(j,jl)  += wlo;
    Dth_odd(j,j)   += wco;
    Dth_odd(j,jr)  += wro;
    Dth_odd(j,jrr) += wrro;
    DDth_even(j,jll) += w2lle;
    DDth_even(j,jl)  += w2le;
    DDth_even(j,j)   += w2ce;
    DDth_even(j,jr)  += w2re;
    DDth_even(j,jrr) += w2rre;
    DDth_odd(j,jll) += w2llo;
    DDth_odd(j,jl)  += w2lo;
    DDth_odd(j,j)   += w2co;
    DDth_odd(j,jr)  += w2ro;
    DDth_odd(j,jrr) += w2rro;
  }
  // step 2. solve for eigenvalues and eigenvectors
  for (int k=0; k<nk2; k++) {
    int im = k+is[2][2];
    // use even only for m=0
    // strictly speaking, all even m should have even eigenvectors P
    // (and for m=2,4,6... the leading order around the pole is th^2)
    // but empirically, using Dth_odd to force P=0 at pole works
    // slightly better than using Dth_even.
    if (im==0) {
      Dth = Dth_even;
      DDth = DDth_even;
    } else {
      Dth = Dth_odd;
      DDth = DDth_odd;
    }
    for (int j=0; j<nth; j++) {
      Real tanth = std::tan(x2(j));
      for (int j2=0; j2<nth; j2++) {
        M(j,j2) = DDth(j,j2) + 1./tanth*Dth(j,j2);
      }
      M(j,j) += dsq_phi_eigen(im)/SQR(std::sin(x2(j)));
    }
    Eigen::EigenSolver<Eigen::MatrixXd> es(M);
    Eigen::MatrixXd V = es.eigenvectors().real(); // these should be all real
    Eigen::VectorXd W = es.eigenvalues().real(); // these should be all real
    // grid to basis: invert V
    Eigen::MatrixXd Vinv = V.inverse();
    // sort the eigenvalues and eigenvectors
    // theoretically, we should get the same result on all cores working on the same m, l.
    std::vector<Real> wvec(nth);
    for (int j=0; j<nth; j++) wvec[j] = std::abs(W[j]);
    std::vector<int> idx = argsort(wvec);
    for (int il=0; il<nth; il++) {
      int il1 = idx[il];
      dsq_thphi_eigen(im,il) = W[il1];
      for (int ith=0; ith<nth; ith++) {
        basis_to_grid(k,ith,il) = V(ith,il1);
        grid_to_basis(k,il,ith) = Vinv(il1,ith);
      }
    }
  }

  // step 3. collect dsq_thphi_eigen form other m ranges
  // (because those might be used in radial initialization)
  #ifdef MPI_PARALLEL
  Real * buf_in = new Real[Nj*Nk];
  Real * buf_out = new Real[Nj*Nk];
  int offset=0;
  BufferUtility::PackData(dsq_thphi_eigen, buf_in, 0, Nj-1, 0, Nk-1, 0, 0, offset);
  MPI_Allreduce(buf_in, buf_out, Nj*Nk, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD); // assumes all eigenvalues are negative
  offset=0;
  BufferUtility::UnpackData(buf_out, dsq_thphi_eigen, 0, Nj-1, 0, Nk-1, 0, 0, offset);
  delete[] buf_in;
  delete[] buf_out;
  #endif
}


//----------------------------------------------------------------------------------------
//! \fn void SphGravity::InitializePhi()
//! \brief set up rfft plans and compute phi eigenvalues

void SphGravity::InitializePhi(){
  // assume uniform phi grid
  int Nk = N[2];
  int ni3=n[3][0], nj3=n[3][1], nk3=n[3][2];
  Real dphi = xf[2](1)-xf[2](0);
  dsq_phi_eigen.NewAthenaArray(Nk);
  dsq_phi_eigen(0) = 0.;
  for (int k=1; k<Nk; ++k) {
    Real dphi_k = ((Real) k)/((Real) Nk)*TWO_PI;
    // 4th order finite difference
    dsq_phi_eigen(k) = (-2.5 + 8./3.*std::cos(dphi_k) - 1./6.*std::cos(2.*dphi_k)) /SQR(dphi);
  }
  // phi plans
  fftw_r2r_kind kind_fwd;
  fftw_r2r_kind kind_bck;
  kind_fwd = FFTW_R2HC;
  kind_bck = FFTW_HC2R;
  // input/output for these plans will be
  // overridden when executed
  r2r_fft_fwd = fftw_plan_many_r2r(1,&nk3,nj3*ni3,
                                   NULL,NULL,1,nk3,
                                   NULL,NULL,1,nk3,
                                   &kind_fwd,FFTW_ESTIMATE);
  r2r_fft_bck = fftw_plan_many_r2r(1,&nk3,nj3*ni3,
                                   NULL,NULL,1,nk3,
                                   NULL,NULL,1,nk3,
                                   &kind_bck,FFTW_ESTIMATE);
}
#endif // SELF_GRAVITY_ENABLED == 3
#endif // FFT