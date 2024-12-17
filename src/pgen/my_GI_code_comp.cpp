#include <iostream>
#include <sstream>
#include <algorithm>    // std::min

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../inputs/hdf5_reader.hpp"  // HDF5ReadRealArray()
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

#include "../gravity/gravity.hpp"
#include "../gravity/sph_gravity.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//========================================================================================
// Parameters
//========================================================================================

const bool is_cart = (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0);
  // assume that the coordinate is either cartesian or spherical

// star
const Real M = 1.;
const Real G = 1.;
const bool fix_star_at_origin = true;
Real Mtot; // star + disk; to be computed after pgen

// disk
Real Rd; // reference radius
Real Rd_in;
Real Rd_out;
Real Sigma_d;
Real Sigma_slope;
Real T_d;
Real T_slope;

// cooling
Real beta_cool;
//Real hypercool_density_threshold=1.e-6;

// grid
Real nth_lo; // # of low/mid res cells (for half pi)
Real nth_hi; // # of high res cells (for half pi)
Real h_hi; // height of high res region (in rad)
Real dth_pole; // cell size at pole
int l0, m0;

// initial perturbation
Real perturbation_relative_amplitude = 0.01;
int perturbation_m_min = 1;
int perturbation_m_max = 6;
int random_seed = 2024; // fix this across simulations to use the same physical ic


//========================================================================================
// Forward declarations
//========================================================================================
void MySource(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
Real MeshGen(Real x, RegionSize rs);
Real MyHst(MeshBlock *pmb, int iout);
void GetStellarMassAndLocation(SphGravity * grav, MeshBlock * pmb);







//========================================================================================
// Utility functions
//========================================================================================

void GetCylCoord(MeshBlock *pmb, int k, int j, int i, Real & R, Real & phi, Real & z){
  Real x1 = pmb->pcoord->x1v(i);
  Real x2 = pmb->pcoord->x2v(j);
  Real x3 = pmb->pcoord->x3v(k);
  if (is_cart) { // cartesian
    R = std::sqrt(x1*x1+x2*x2);
    phi = std::atan2(x2,x1);
    z = x3;
  } else { // spherical
    R = x1*std::sin(x2);
    phi = x3;
    z = x1*std::cos(x2);
  }
  return;
}
void NphiToN123(MeshBlock *pmb, int k, int j, int i, Real & n1, Real & n2, Real & n3){
  // unit vector in phi -> components in 123
  if (is_cart) { // cartesian
    Real x1 = pmb->pcoord->x1v(i);
    Real x2 = pmb->pcoord->x2v(j);
    Real R = std::sqrt(x1*x1+x2*x2);
    n1 = -x2/R;
    n2 = x1/R;
    n3 = 0.;
  } else { // spherical
    n1 = 0.;
    n2 = 0.;
    n3 = 1.;
  }
}





//========================================================================================
// Initialize user data
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // star: sanity check
  if (pin->GetOrAddReal("problem","GM",0.)!=0.) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function Mesh::InitUserMeshData"
        << std::endl << "don't set GM input file!" <<std::endl;
    ATHENA_ERROR(msg);
  }

  // disk
  Rd          = pin->GetReal("problem","Rd");
  Rd_in       = pin->GetReal("problem","Rd_in");
  Rd_out      = pin->GetReal("problem","Rd_out");
  Sigma_d     = pin->GetReal("problem","Sigma_d");
  Sigma_slope = pin->GetReal("problem","Sigma_slope");
  T_d         = pin->GetReal("problem","T_d");
  T_slope     = pin->GetReal("problem","T_slope");

  // cooling
  beta_cool = pin->GetReal("problem","beta_cool");


  // grid
  if (pin->GetOrAddReal("mesh","x2rat",1.0)<0.) {
    nth_lo   = pin->GetReal("mesh","nth_lo");
    nth_hi   = pin->GetReal("mesh","nth_hi");
    h_hi     = pin->GetReal("mesh","h_hi");
    dth_pole = pin->GetReal("mesh","dth_pole");
    EnrollUserMeshGenerator(X2DIR,MeshGen);
  }
  l0 = std::round(PI/(pin->GetReal("mesh","x2max")-pin->GetReal("mesh","x2min")));
  m0 = std::round(2.*PI/(pin->GetReal("mesh","x3max")-pin->GetReal("mesh","x3min")));

  // perturbation
  perturbation_relative_amplitude = pin->GetOrAddReal("problem","perturbation_relative_amplitude",perturbation_relative_amplitude);
  perturbation_m_min = pin->GetOrAddInteger("problem","perturbation_m_min",perturbation_m_min);
  perturbation_m_max = pin->GetOrAddInteger("problem","perturbation_m_max",perturbation_m_max);
  random_seed = pin->GetOrAddInteger("problem","random_seed",random_seed);

  // physics
  SetFourPiG(4.*PI*G);
  // confirm that we haven't declared G in input
  if (pin->DoesParameterExist("problem","four_pi_G")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "this problem generator uses a fixed G=1; do not set four_pi_G in input!";
    ATHENA_ERROR(msg);
  }

  // source (cooling and relaxation)
  EnrollUserExplicitSourceFunction(MySource);

  // boundary conitions
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }

  // hst outputs
  AllocateUserHistoryOutput(4); // rho_max, rho_rel_max, T_max, v_max
  EnrollUserHistoryOutput(0, MyHst, "rho_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, MyHst, "rho_rel_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, MyHst, "T_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, MyHst, "v_max", UserHistoryOperation::max);
    // rho_rel is relative to (extrapolated) initial midplane density

  AllocateRealUserMeshDataField(1);
  ruser_mesh_data[0].NewAthenaArray(4); // stellar properties
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // enroll function for updating the star
  // do this here because gravity driver is initialized after InitUserMeshData()
  if (!is_cart) pmy_mesh->psgrd->EnrollUpdateStarFn(GetStellarMassAndLocation);
}







//========================================================================================
// Problem generator
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const Real gamma = peos->GetGamma();
  // initial perturbations
  std::srand(random_seed); // fix the random seed on each process...
  int m_min = perturbation_m_min, m_max = perturbation_m_max;
  Real * amplitudes = new Real [m_max-m_min+1];
  Real * phases = new Real [m_max-m_min+1];
  for (int m=m_min; m<m_max; ++m) {
    amplitudes[m-m_min] = perturbation_relative_amplitude;
    phases[m-m_min] = ((Real) std::rand()/RAND_MAX) * 2.*PI;
  }
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real R, phi, z;
        GetCylCoord(this, k, j, i, R, phi, z);
        Real Sigma = (R<=Rd_out && R>Rd_in) ? Sigma_d * std::pow(R/Rd, Sigma_slope) : 0.;
        Real T = T_d * std::pow(R/Rd, T_slope);
        Real H = std::sqrt(T) / std::sqrt(G*M/(R*R*R)); // H = cs_iso/Omega
        Real rho_mid = Sigma/H/std::sqrt(2.*PI);
        Real r = std::sqrt(R*R+z*z);
        Real MdR = 2.*PI * Sigma_d * std::max(0.,std::log(R/Rd_in)); // assuming slope=-2 and Rd=1...
        Real vphi = (R<=Rd_out && R>Rd_in) ? std::sqrt(G*(M+MdR)*R*R/(r*r*r)) : 0.;
        Real factor = 1.;
        for (int m=m_min; m<m_max; ++m) {
          factor += std::cos(m*(phi+phases[m-m_min])) * amplitudes[m-m_min];
        }
        Real n1, n2, n3;
        NphiToN123(this, k, j, i, n1, n2, n3);
        phydro->u(IDN,k,j,i) = std::max(1.e-16, factor * rho_mid * std::exp(-.5*SQR(z/H))); // just a hard-coded density floor
        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i) * vphi * n1;
        phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i) * vphi * n2;
        phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i) * vphi * n3;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)*T/(gamma-1.)
           + .5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
      }
    }
  }
  Mtot = M + 2.*PI * Sigma_d * std::max(0.,std::log(Rd_out/Rd_in)); // assuming slope=-2 and Rd=1...
}





//========================================================================================
// Source term: cooling
//========================================================================================
void MySource(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  const Real gamma = pmb->peos->GetGamma();
  const Real gm1 = pmb->peos->GetGamma()-1.;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real R, phi, z;
        GetCylCoord(pmb, k, j, i, R, phi, z);
        Real temp = prim(IPR,k,j,i) / prim(IDN,k,j,i);
        Real cooling_rate = std::sqrt(G*M/R/R/R)/beta_cool; // beta cooling
        //cooling_rate *= (1.+hypercool_density_threshold/prim(IDN,k,j,i));
        cons(IEN,k,j,i) -= prim(IPR,k,j,i) / gm1 * (1-std::exp(-dt*cooling_rate));
        // enforce a temperature cap
        Real T_cap = 1;
        Real Ek = .5*(SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) + SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
        Real E_cap = cons(IDN,k,j,i)*T_cap/gm1 + Ek;
        if (cons(IEN,k,j,i)>E_cap) cons(IEN,k,j,i)=E_cap;
      }
    }
  }
}







//========================================================================================
// Boundary conditions
//========================================================================================
// this applies only to spherical coord, since for cartesian we can just use built-in bcs
// reflect poloidal velocity, maintain rotation
void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real r1=pmb->pcoord->x1v(iu+i), r2=pmb->pcoord->x1v(iu-i+1);
        prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu-i+1);
        prim(IVX,k,j,iu+i) = -prim(IVX,k,j,iu-i+1)*SQR(r2)/SQR(r1);
        prim(IVY,k,j,iu+i) = -prim(IVY,k,j,iu-i+1);
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu-i+1)*r1/r2;
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu-i+1);
      }
    }
  }
}
// outflow + velocity cap
void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real r1=pmb->pcoord->x1v(il-i), r2=pmb->pcoord->x1v(il+i-1);
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il+i-1);
        prim(IVX,k,j,il-i) = std::min(0.,prim(IVX,k,j,il+i-1)*SQR(r2)/SQR(r1));
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il+i-1);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il+i-1)*r1/r2;
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il+i-1);
      }
    }
  }
}









//========================================================================================
// Mesh generator
//========================================================================================
Real MeshGen(Real x, RegionSize rs) {
  // variables set externally: nth_lo, nth_hi, h_hi, dth_pole

  // sanity check: is the resolution correct?
  bool reflect = (std::abs(rs.x2max-.5*PI)<1.e-8);
  if ((2-reflect) * (nth_lo + nth_hi) != rs.nx2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshGen" << std::endl
        << "input nth_lo and nth_hi do not agree with grid";
    ATHENA_ERROR(msg);
  }

  Real dth_mid = h_hi/nth_hi;
  Real A = PI*.5-h_hi;
  Real gamma = (dth_pole*std::log(dth_pole/dth_mid)+dth_mid-dth_pole) / (nth_lo*dth_pole-A);
  
  // remap to x=0 at midplane and 1 at pole
  Real sign = 1.;
  if (!reflect) {
    if (x>.5) sign = -1.;
    x = std::abs(x-.5) * 2.;
  } else {
    x = 1.-x;
  }
  // remap to x = nth_lo+nth_hi at pole
  x *= nth_lo+nth_hi;
  // y: 0 at midplane and pi/2 at pole
  Real y;
  if (x<=nth_hi) y = x*dth_mid;
  else if (x<=nth_hi+std::log(dth_pole/dth_mid)/gamma) y = nth_hi*dth_mid + 1./gamma*(std::exp((x-nth_hi)*gamma)-1.)*dth_mid;
  else y = PI*.5 - dth_pole*(nth_hi+nth_lo-x);
  // remap y
  y /= (PI*.5);
  if (!reflect) {
    y = .5-.5*y*sign;
  } else {
    y = 1.-y;
  }
  return y*rs.x2max + (1.-y)*rs.x2min;
}









//========================================================================================
// History output
//========================================================================================
Real MyHst(MeshBlock *pmb, int iout){
  // max density, relative density, temperature, or velocity
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  Real y_max = 0.;
  const Real gamma = pmb->peos->GetGamma();
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        Real y;
        if (iout==0) {y = pmb->phydro->u(IDN,k,j,i);}
        else if (iout==1) {
          y = pmb->phydro->u(IDN,k,j,i);
          // find rho_mid; this is similar to what we do in ProblemGenerator()
          Real R, phi, z;
          GetCylCoord(pmb, k, j, i, R, phi, z);
          Real Sigma = Sigma_d * std::pow(R/Rd, Sigma_slope);
          Real T = T_d * std::pow(R/Rd, T_slope);
          Real H = std::sqrt(T) / std::sqrt(G*M/(R*R*R));
          Real rho_mid = Sigma/H/std::sqrt(2.*PI);
          y /= rho_mid;
        }
        else if (iout==2) {
          y = pmb->phydro->w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i);
        }
        else if (iout==3) {
          y = std::sqrt(SQR(pmb->phydro->w(IVX,k,j,i))+SQR(pmb->phydro->w(IVY,k,j,i))+SQR(pmb->phydro->w(IVZ,k,j,i)));
        }
        if (y>y_max) y_max = y;
      }
    }
  }
  return y_max;
}



//========================================================================================
// Get stellar properties (for spherical only)
//========================================================================================
void GetStellarMassAndLocation(SphGravity * grav, MeshBlock * pmb) {
  Real M [4] = {0.,0.,0.,0.};
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real r = pmb->pcoord->x1v(i);
        Real th = pmb->pcoord->x2v(j);
        Real phi = pmb->pcoord->x3v(k);
        Real x = r*std::sin(th)*std::cos(phi);
        Real y = r*std::sin(th)*std::sin(phi);
        Real z = r*std::cos(th);
        Real vol = pmb->pcoord->GetCellVolume(k,j,i);
        Real den = pmb->phydro->u(IDN,k,j,i);
        M[0] += den*vol;
        M[1] += den*vol*x;
        M[2] += den*vol*y;
        M[3] += den*vol*z;
      }
    }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &M, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  M[0]*=m0*l0;
  M[1]*=m0*l0;
  M[2]*=m0*l0;
  M[3]*=m0*l0;
  Real M_star = Mtot - M[0];
  Real x_star = fix_star_at_origin ? 0. : -M[1]/M_star;
  Real y_star = fix_star_at_origin ? 0. : -M[2]/M_star;
  Real z_star = fix_star_at_origin ? 0. : -M[3]/M_star;

  // correction for 2d and 1d
  if (pmb->block_size.nx3==1) {
    x_star = 0.;
    y_star = 0.;
    if (pmb->block_size.nx2==1) z_star = 0.;
    // physically we don't expect this 1d case to be ever relevant...
    // but keep this here just in case
  }

  // save results in ruser_mesh_data
  pmb->pmy_mesh->ruser_mesh_data[0](0) = M_star;
  pmb->pmy_mesh->ruser_mesh_data[0](1) = x_star;
  pmb->pmy_mesh->ruser_mesh_data[0](2) = y_star;
  pmb->pmy_mesh->ruser_mesh_data[0](3) = z_star;
  // save results in grav
  grav->M_star = M_star;
  grav->x_star = x_star;
  grav->y_star = y_star;
  grav->z_star = z_star;
}