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

#include "../gravity/gravity.hpp"
#include "../gravity/sph_gravity.hpp"

#include "../standalone_physics/optical_depth_rth.hpp"

#include "../nr_radiation/radiation.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif









//========================================================================================
// Parameters - fixed
//========================================================================================
const Real Mtot = 1.; // total mass (disk+star)
const Real G = 1.; // do not read four_pi_G from input

Real Sigma_slope = -1.99; // disk surface density slope
  // this should be -2, but keeping it !=-2 avoid divergences in calculations
Real T_slope = -1; // temperature slope
// these are fixed to give approximately uniform Q and h/r thoughout the disk

//========================================================================================
// Parameters - read from input
//========================================================================================

// grid

Real nth_lo; // # of low/mid res cells (for half pi)
Real nth_hi; // # of high res cells (for half pi)
Real h_hi; // height of high res region (in rad)
Real dth_pole; // cell size at pole

// initial conditions

bool read_from_2d = false; // read 2d initial conditions from athdf file (assume cons & single meshblock)
bool read_from_3d = false; // read 3d initial conditions from athdf file (assume prim & same meshblock layout as current sim)
std::string input_filename; // athdf file containing IC

Real Rd; // outer edge of disk
Real Rd_in; // inner edge of disk
Real Md; // total disk mass - this indirectly controls h/r
Real Qd; // initial Q at Rd (defined using adiabatic sound speed and Keplerian kappa)

// star

bool fix_star_at_origin = false; // fix the star at origin

// cooling

int cooling_mode = 0;
  // 0: cooling off (or use radiation)
  // 1: linear cooling: t_cool = beta_cool / OmegaK

Real beta_cool;
Real hypercool_density_threshold = 1.e-8;
  // threshold for hypercooling - we multiplty cooling rate by
  // (1+hypercool_density_threshold/density)
  // <0 to turn off hypercooling at low density

// radiation

Real kappa = 0.;

// radiation boundary condition: true for vacuum, false for outflow

bool vacuum_inner_x1 = false;
bool vacuum_outer_x1 = false;

// relaxation (used for making ic)

bool relaxation = false;

// initial perturbation:
// a range of different m with the same amplitude and random phases

bool inject_perturbation=false;
Real perturbation_relative_amplitude;
int perturbation_m_min;
int perturbation_m_max;
int random_seed = 2023; // fix this across simulations to use the same physical ic









//========================================================================================
// Forward declarations
//========================================================================================

void MySource(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void RadInnerX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad,
                const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh);
void RadOuterX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad,
                const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh);

void GetStellarMassAndLocation(SphGravity * grav, MeshBlock * pmb);

Real MeshGen(Real x, RegionSize rs);

Real MyHst(MeshBlock *pmb, int iout);







//========================================================================================
// Initialize user data
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

  // read parameters

  // grid
  if (pin->GetOrAddReal("mesh","x2rat",1.0)<0.) {
    nth_lo   = pin->GetReal("mesh","nth_lo");
    nth_hi   = pin->GetReal("mesh","nth_hi");
    h_hi     = pin->GetReal("mesh","h_hi");
    dth_pole = pin->GetReal("mesh","dth_pole");
  }
  // initial conditions
  read_from_2d = pin->GetOrAddBoolean("problem","read_from_2d",read_from_2d);
  read_from_3d = pin->GetOrAddBoolean("problem","read_from_3d",read_from_3d);
  if (read_from_2d || read_from_3d)
    input_filename = pin->GetString("problem", "input_filename");
  Rd    = pin->GetReal("problem","Rd");
  Rd_in = pin->GetReal("problem","Rd_in");
  Md    = pin->GetReal("problem","Md");
  Qd    = pin->GetReal("problem","Qd");
  // star
  fix_star_at_origin = pin->GetOrAddBoolean("problem","fix_star_at_origin",fix_star_at_origin);
  // cooling
  cooling_mode = pin->GetOrAddInteger("problem","cooling_mode",cooling_mode);
  if (cooling_mode==1) {
    beta_cool = pin->GetReal("problem","beta_cool");
    hypercool_density_threshold = pin->GetOrAddReal("problem","hypercool_density_threshold",hypercool_density_threshold);
  }
  // radiation
  kappa = pin->GetOrAddReal("problem","kappa",kappa);
  // radiation boundary
  vacuum_inner_x1 = pin->GetOrAddBoolean("problem","vacuum_inner_x1",vacuum_inner_x1);
  vacuum_outer_x1 = pin->GetOrAddBoolean("problem","vacuum_outer_x1",vacuum_outer_x1);
  // relaxation (used for making ic)
  relaxation = pin->GetOrAddBoolean("problem","relaxation",relaxation);
  // initial perturbation
  inject_perturbation = pin->GetOrAddBoolean("problem","inject_perturbation",inject_perturbation);
  if (inject_perturbation) {
    perturbation_relative_amplitude = pin->GetReal("problem","perturbation_relative_amplitude");
    perturbation_m_min = pin->GetInteger("problem","perturbation_m_min");
    perturbation_m_max = pin->GetInteger("problem","perturbation_m_max");
    random_seed = pin->GetOrAddInteger("problem","random_seed",random_seed);
  }

  // set physics
  
  // gravity
  SetFourPiG(4.*PI*G);
  // confirm that we haven't declared G in input
  if (pin->DoesParameterExist("problem","four_pi_G")) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "this problem generator uses a fixed G; do not set four_pi_G in input!";
    ATHENA_ERROR(msg);
  }
  // source (cooling and relaxation)
  EnrollUserExplicitSourceFunction(MySource);
  // boundary conitions
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
    if (NR_RADIATION_ENABLED||IM_RADIATION_ENABLED) EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, RadInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
    if (NR_RADIATION_ENABLED||IM_RADIATION_ENABLED) EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, RadOuterX1);
  }
  // mesh generator
  if (pin->GetOrAddReal("mesh","x2rat",1.0)<0.)
    EnrollUserMeshGenerator(X2DIR,MeshGen);

  // data field

  AllocateRealUserMeshDataField(1);
  ruser_mesh_data[0].NewAthenaArray(4); // stellar properties

  // hst outputs

  AllocateUserHistoryOutput(8); // Mstar, xstar, ystar, zstar, rho_max, rho_rel_max, T_max, v_max
  EnrollUserHistoryOutput(0, MyHst, "Mstar", UserHistoryOperation::max);
  EnrollUserHistoryOutput(1, MyHst, "xstar", UserHistoryOperation::max);
  EnrollUserHistoryOutput(2, MyHst, "ystar", UserHistoryOperation::max);
  EnrollUserHistoryOutput(3, MyHst, "zstar", UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, MyHst, "rho_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(5, MyHst, "rho_rel_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, MyHst, "T_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(7, MyHst, "v_max", UserHistoryOperation::max);
    // rho_rel is relative to (extrapolated) initial midplane density
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // enroll function for updating the star
  // do this here because gravity driver is initialized after InitUserMeshData()
  pmy_mesh->psgrd->EnrollUpdateStarFn(GetStellarMassAndLocation);

  if(NR_RADIATION_ENABLED||IM_RADIATION_ENABLED) pnrrad->EnrollOpacityFunction(DiskOpacity);
#ifdef RAD_ITR_DIAGNOSTICS
  AllocateUserOutputVariables(10*2); // difference and Er for each iteration
#endif
#ifdef SAVE_HEATING_RATE
  AllocateUserOutputVariables(1); // heating rate
#endif
}









//========================================================================================
// Problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (read_from_2d) { // case 1. read from 2d prim athdf file with same resolution and single meshblock
    std::string dataset_cons = "cons";
    // make a global array
    AthenaArray<Real> u_global;
    int nx1_global = pmy_mesh->mesh_size.nx1 + 2*NGHOST;
    int nx2_global = pmy_mesh->mesh_size.nx2 + 2*NGHOST;
    int nx3_global = 1;
    u_global.NewAthenaArray(NHYDRO,nx3_global,nx2_global,nx1_global);
    // read data
    int start_cons_file[5];
    start_cons_file[1] = 0; // gid
    start_cons_file[2] = 0;
    start_cons_file[3] = 0;
    start_cons_file[4] = 0;
    int start_cons_indices[5];
    start_cons_indices[IDN] = 0;
    start_cons_indices[IM1] = 2;
    start_cons_indices[IM2] = 3;
    start_cons_indices[IM3] = 4;
    start_cons_indices[IEN] = 1;
    int count_cons_file[5];
    count_cons_file[0] = 1;
    count_cons_file[1] = 1;
    count_cons_file[2] = 1;
    count_cons_file[3] = pmy_mesh->mesh_size.nx2;
    count_cons_file[4] = pmy_mesh->mesh_size.nx1;
    int start_cons_mem[4];
    start_cons_mem[1] = 0;
    start_cons_mem[2] = js;
    start_cons_mem[3] = is;
    int count_cons_mem[4];
    count_cons_mem[0] = 1;
    count_cons_mem[1] = 1;
    count_cons_mem[2] = pmy_mesh->mesh_size.nx2;
    count_cons_mem[3] = pmy_mesh->mesh_size.nx1;
    for (int n = 0; n < NHYDRO; ++n) {
      start_cons_file[0] = start_cons_indices[n];
      start_cons_mem[0] = n;
      HDF5ReadRealArray(input_filename.c_str(), dataset_cons.c_str(), 5, start_cons_file,
                        count_cons_file, 4, start_cons_mem,
                        count_cons_mem, u_global, true);
    }
    // dump data into grid
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          int kg, jg, ig;
          kg = 0;
          jg = j + block_size.nx2*loc.lx2;
          ig = i + block_size.nx1*loc.lx1;
          for (int n = 0; n < NHYDRO; ++n)
            phydro->u(n,k,j,i) = u_global(n,kg,jg,ig);
        }
      }
    }
    // delete scratch array
    u_global.DeleteAthenaArray();
    // radiation
    if (NR_RADIATION_ENABLED||IM_RADIATION_ENABLED) pnrrad->ir.ZeroClear();
  }
    
  else if (read_from_3d) { // case 2. read from 3d prim athdf file with same resolution and meskblock layout
    std::string dataset_prim = "prim";
    int start_cons_file[5];
    start_cons_file[1] = gid; // assume same mb layout
    start_cons_file[2] = 0;
    start_cons_file[3] = 0;
    start_cons_file[4] = 0;
    int start_cons_indices[5];
    start_cons_indices[IDN] = 0;
    start_cons_indices[IM1] = 2; // actually ivx
    start_cons_indices[IM2] = 3; // actually ivy
    start_cons_indices[IM3] = 4; // actually ivz
    start_cons_indices[IEN] = 1; // actually ipr
    int count_cons_file[5];
    count_cons_file[0] = 1;
    count_cons_file[1] = 1;
    count_cons_file[2] = block_size.nx3;
    count_cons_file[3] = block_size.nx2;
    count_cons_file[4] = block_size.nx1;
    int start_cons_mem[4];
    start_cons_mem[1] = ks;
    start_cons_mem[2] = js;
    start_cons_mem[3] = is;
    int count_cons_mem[4];
    count_cons_mem[0] = 1;
    count_cons_mem[1] = block_size.nx3;
    count_cons_mem[2] = block_size.nx2;
    count_cons_mem[3] = block_size.nx1;

    // Set conserved values from file
    for (int n = 0; n < NHYDRO; ++n) {
      start_cons_file[0] = start_cons_indices[n];
      start_cons_mem[0] = n;
      HDF5ReadRealArray(input_filename.c_str(), dataset_prim.c_str(), 5, start_cons_file,
                        count_cons_file, 4, start_cons_mem,
                        count_cons_mem, phydro->u, true);
    }

    // update conserved value (because u currently stores w)
    const Real gamma = peos->GetGamma();
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          phydro->u(IM1,k,j,i) *= phydro->u(IDN,k,j,i);
          phydro->u(IM2,k,j,i) *= phydro->u(IDN,k,j,i);
          phydro->u(IM3,k,j,i) *= phydro->u(IDN,k,j,i);
          if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN,k,j,i) = phydro->u(IEN,k,j,i)/(gamma-1.)
             + .5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
          }
        }
      }
    }

    // radiation
    if (NR_RADIATION_ENABLED||IM_RADIATION_ENABLED) pnrrad->ir.ZeroClear();
  }

  else { // case 3. make new initial condition
    const Real gamma = peos->GetGamma();
    Real Sigma_d = Md/(2.*PI*Rd*Rd)/(1-std::pow(Rd_in/Rd, 2+Sigma_slope))*(2+Sigma_slope); // Sigma at Rd
    Real T_d = SQR(PI*G*Sigma_d*Qd/std::sqrt(G*Mtot/Rd/Rd/Rd))/gamma; // T at Rd
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real R, z;

          Real r = pcoord->x1v(i);
          Real th = pcoord->x2v(j);
          R = r*std::sin(th);
          z = r*std::cos(th);

          Real Sigma = (R<=Rd && R>Rd_in) ? Sigma_d * std::pow(R/Rd, Sigma_slope) : 0.;
          Real T = T_d * std::pow(R/Rd, T_slope);
          Real MR = Mtot; 
          if (R<Rd) MR = Mtot - 2.*PI * Sigma_d * (1.-std::pow(R/Rd, 2+Sigma_slope)) / (2+Sigma_slope); // M(<R)
          Real H = std::sqrt(T) / std::sqrt(G*MR/(R*R*R)); // computed using isothermal sound speed cs_iso = sqrt(T)
          Real rho_mid = Sigma/H/std::sqrt(2.*PI);
          phydro->u(IDN,k,j,i) = rho_mid * std::exp(-.5*SQR(z/H));
          phydro->u(IM1,k,j,i) = 0.;
          phydro->u(IM2,k,j,i) = 0.;
          phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i) * std::sqrt(R*R/(r*r*r));
          if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)*T/(gamma-1.)
             + .5*(SQR(phydro->u(IM1,k,j,i)) + SQR(phydro->u(IM2,k,j,i)) + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
          }
        }
      }
    }
    
    // radiation
    if (NR_RADIATION_ENABLED||IM_RADIATION_ENABLED) pnrrad->ir.ZeroClear();
  }
}









//========================================================================================
// Source term: cooling and relaxation
//========================================================================================

void MySource(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  const Real gamma = pmb->peos->GetGamma();
  const Real gm1 = pmb->peos->GetGamma()-1.;
  if (relaxation) { // case 1. relaxation
    Real Sigma_d = Md/(2.*PI*Rd*Rd)/(1-std::pow(Rd_in/Rd, 2+Sigma_slope))*(2+Sigma_slope); // Sigma at Rd
    Real T_d = SQR(PI*G*Sigma_d*Qd/std::sqrt(G*Mtot/Rd/Rd/Rd))/gamma; // T at Rd
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r = pmb->pcoord->x1v(i);
          Real th = pmb->pcoord->x2v(j);
          Real R = r*std::sin(th);
          Real p_goal = prim(IDN,k,j,i) * T_d * std::pow(R/Rd, T_slope);
          Real v = std::sqrt(SQR(prim(IVX,k,j,i))+SQR(prim(IVY,k,j,i)));
          Real vK = std::sqrt(G*Mtot/R);
          Real cooling_rate = 10. * std::sqrt(G*Mtot/R/R/R);
          cooling_rate *= (1.+hypercool_density_threshold/prim(IDN,k,j,i));
          Real damping_rate = 10. * std::sqrt(G*Mtot/R/R/R) * std::max(1., v/(0.01*vK));
          cons(IEN,k,j,i) -= (prim(IPR,k,j,i)-p_goal) / gm1 * (1-std::exp(-dt*cooling_rate));
          cons(IM1,k,j,i) -= prim(IDN,k,j,i) * prim(IVX,k,j,i) * (1.-std::exp(-dt*damping_rate));
          cons(IM2,k,j,i) -= prim(IDN,k,j,i) * prim(IVY,k,j,i) * (1.-std::exp(-dt*damping_rate));
        }
      }
    }
  }
  else if (cooling_mode==1) { // case 2. beta cooling
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          Real r = pmb->pcoord->x1v(i);
          Real th = pmb->pcoord->x2v(j);
          Real R = r*std::sin(th);
          Real temp = prim(IPR,k,j,i) / prim(IDN,k,j,i);
          Real cooling_rate = std::sqrt(G*Mtot/R/R/R)/beta_cool; // beta cooling
          cooling_rate *= (1.+hypercool_density_threshold/prim(IDN,k,j,i));
          cons(IEN,k,j,i) -= prim(IPR,k,j,i) / gm1 * (1-std::exp(-dt*cooling_rate));
        }
      }
    }
  }
}









//========================================================================================
// Opacity function: constant opacity
//========================================================================================

void DiskOpacity(MeshBlock *pmb, AthenaArray<Real> &prim) {
  NRRadiation *pnrrad = pmb->pnrrad;
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;
  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real sigma = prim(IDN,k,j,i) * kappa;
        for (int ifr=0; ifr<pnrrad->nfreq; ++ifr){
          pnrrad->sigma_s(k,j,i,ifr) = 0.0;
          pnrrad->sigma_a(k,j,i,ifr) = sigma;
          pnrrad->sigma_pe(k,j,i,ifr) = sigma;
          pnrrad->sigma_p(k,j,i,ifr) = sigma;
        }
      }
    }
  }
}









//========================================================================================
// User work in loop: inject perturbation at the end of the first timestep
//========================================================================================

void MeshBlock::UserWorkInLoop() {
  // inject perturbation
  if (inject_perturbation) {
    inject_perturbation = false;
    if (Globals::my_rank == 0) {
      std::cout<<"Injecting perturbation..."<<std::endl;
    }
    std::srand(random_seed); // fix the random seed on each process...
    int m_min = perturbation_m_min, m_max = perturbation_m_max;
    Real * amplitudes = new Real [m_max-m_min+1];
    Real * phases = new Real [m_max-m_min+1];
    for (int m=m_min; m<m_max; ++m) {
      amplitudes[m-m_min] = perturbation_relative_amplitude;
      phases[m-m_min] = ((Real) std::rand()/RAND_MAX) * 2.*PI;
    }
    Real dfloor = peos->GetDensityFloor();
    for (int k=ks; k<=ke; ++k) {
      Real phi = pcoord->x3v(k);
      Real factor = 1.;
      for (int m=m_min; m<m_max; ++m) {
        factor += std::cos(m*(phi+phases[m-m_min])) * amplitudes[m-m_min];
      }
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          // leave dfloor fixed            
          if (phydro->u(IDN,k,j,i)>dfloor) {
            Real factor_d = (phydro->u(IDN,k,j,i)-dfloor)/phydro->u(IDN,k,j,i)*(factor-1.) + 1.;
            // perturb density, fix velocity and temperature
            phydro->u(IDN,k,j,i) *= factor_d;
            phydro->u(IM1,k,j,i) *= factor_d;
            phydro->u(IM2,k,j,i) *= factor_d;
            phydro->u(IM3,k,j,i) *= factor_d;
            if (NON_BAROTROPIC_EOS)
              phydro->u(IEN,k,j,i) *= factor_d;
          }
        }
      }
    }
    delete[] amplitudes;
    delete[] phases;
    if (Globals::my_rank == 0) {
      std::cout<<"Finished injecting perturbation."<<std::endl;
    }
  }
}









//========================================================================================
// Boundary conditions
//========================================================================================

// outer: reflect poloidal velocity, maintain rotation
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
        prim(IVZ,k,j,iu+i) = prim(IVZ,k,j,iu-i+1)*r2/r1;
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu-i+1);
      }
    }
  }
}

// inner: modified outflow
void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt,
                 int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        Real r1=pmb->pcoord->x1v(il-i), r2=pmb->pcoord->x1v(il+i-1);
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il+i-1);
        prim(IVX,k,j,il-i) = std::min(0.,prim(IVX,k,j,il+i-1)*SQR(r2)/SQR(r1));
          // outflow with velocity cap
        prim(IVY,k,j,il-i) = prim(IVY,k,j,il+i-1);
        prim(IVZ,k,j,il-i) = prim(IVZ,k,j,il+i-1)*r2/r1;
          // constant rotation; this avoids extracting angular momentum from inner boundary,
          // which might excite disk eccentricity
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il+i-1);
      }
    }
  }
}

// r boundary: vacuum
void RadInnerX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad,
                const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for(int ifr=0; ifr<pnrrad->nfreq; ++ifr) {
          for(int n=0; n<pnrrad->nang; ++n) {
            Real mu = pnrrad->mu(0,k,j,is-i,ifr*pnrrad->nang+n);
            if (!vacuum_inner_x1 || mu<0.) // flowing out - continuous
              ir(k,j,is-i,ifr*pnrrad->nang+n) = ir(k,j,is-i+1,ifr*pnrrad->nang+n);
            else // flowing in - zero
              ir(k,j,is-i,ifr*pnrrad->nang+n) = 0.0;
          }
        }
      }
    }
  }
}
void RadOuterX1(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad,
                const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir,
                Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for(int ifr=0; ifr<pnrrad->nfreq; ++ifr) {
          for(int n=0; n<pnrrad->nang; ++n) {
            Real mu = pnrrad->mu(0,k,j,ie+i,ifr*pnrrad->nang+n);
            if (!vacuum_outer_x1 || mu>0.) // flowing out - continuous
              ir(k,j,ie+i,ifr*pnrrad->nang+n) = ir(k,j,ie+i-1,ifr*pnrrad->nang+n);
            else // flowing in - zero
              ir(k,j,ie+i,ifr*pnrrad->nang+n) = 0.0;
          }
        }
      }
    }
  }
}







    
    
//========================================================================================
// Get stellar properties
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
  // stellar properties
  if (iout<4) return pmb->pmy_mesh->ruser_mesh_data[0](iout);
  // max density, relative density, temperature, or velocity
  else if (iout==4 || iout==5 || iout==6 || iout==7) {
    int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
    Real y_max = 0.;
    const Real gamma = pmb->peos->GetGamma();
    Real Sigma_d = Md/(2.*PI*Rd*Rd)/(1-std::pow(Rd_in/Rd, 2+Sigma_slope))*(2+Sigma_slope); // Sigma at Rd
    Real T_d = SQR(PI*G*Sigma_d*Qd/std::sqrt(G*Mtot/Rd/Rd/Rd))/gamma; // T at Rd
    for(int k=ks; k<=ke; k++) {
      for(int j=js; j<=je; j++) {
        for(int i=is; i<=ie; i++) {
          Real y;
          if (iout==4) {y = pmb->phydro->u(IDN,k,j,i);}
          else if (iout==5) {
            y = pmb->phydro->u(IDN,k,j,i);
            // find rho_mid; this is similar to what we do in ProblemGenerator()
            Real r = pmb->pcoord->x1v(i);
            Real th = pmb->pcoord->x2v(j);
            Real R = r*std::sin(th);
            Real Sigma = Sigma_d * std::pow(R/Rd, Sigma_slope);
            Real T = T_d * std::pow(R/Rd, T_slope);
            Real H = std::sqrt(T) / std::sqrt(G*Mtot/(R*R*R));
            Real rho_mid = Sigma/H/std::sqrt(2.*PI);
            y /= rho_mid;
          }
          else if (iout==6) {
            y = pmb->phydro->w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i);
          }
          else if (iout==7) {
            y = std::sqrt(SQR(pmb->phydro->w(IVX,k,j,i))+SQR(pmb->phydro->w(IVY,k,j,i))+SQR(pmb->phydro->w(IVZ,k,j,i)));
          }
          if (y>y_max) y_max = y;
        }
      }
    }
    return y_max;
  }
  else return 0.;
}
