///======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min
#include <cstdlib>    // srand

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"
#include "../utils/utils.hpp"
#include "../globals.hpp"


// The space for opacity table

static AthenaArray<Real> opacitytable;
static AthenaArray<Real> planckopacity;
static AthenaArray<Real> logttable;
static AthenaArray<Real> logrhottable;
static AthenaArray<Real> logttable_planck;
static AthenaArray<Real> logrhottable_planck;

static AthenaArray<Real> ini_profile; 
static AthenaArray<Real> rloc;
static AthenaArray<Real> bd_data;



// The global variable

static Real den0, p0_over_r0, mu, midprat;

static Real rhounit;
static Real tunit;
static Real lunit;
static Real tfloor;
static Real rhofloor;

static Real grav0 = 53.78125253538228;
static Real massbottom = 63450429.421584405;
static Real lbottom=14.99641603037;
static int in_line=104000;
static int x1length;
static Real rmax=46.72944917890;
static Real gm;
//sanity check: gm = grav0*lbottom*lbottom

static Real gm_com;
static Real rm2; // binary separation
// gravitational softing as the companion may be in the grid
//static Real rsoft =200.0;
static Real omegarot;
static int n_user_var=4;

//======================================================================================
/*! \file globaldisk.cpp
 *  \brief global accretion disk problem with radiation
 *
 *====================================================================================*/


int RefinementCondition(MeshBlock *pmb); 


void Inflow_X1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void Outflow_X2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void Outflow_rad_X2(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad, 
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void Inflow_rad_X1(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad, 
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void StarOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);

void rossopacity(const Real rho, const Real tgas, Real &kappa, Real &kappa_planck);

void GravityPotential(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar); 

static void SteadyOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh); //disk outer boundary condition


static void FixRadOuterX1new(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad, 
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh); 
      //disk outer boundary condition


static void VelProfile(const Real x1, const Real x2, const Real x3,
                                const Real den, Real &v1, Real &v2, Real &v3);

static void VelProfilesf(const Real x1, const Real x2, const Real x3,
                                const Real den, Real &v1, Real &v2, Real &v3);

// Functions for coordinate conversion
void ConvCarSph(const Real x, const Real y, const Real z, Real &rad, Real &theta, Real &phi);
void ConvSphCar(const Real rad, const Real theta, const Real phi, Real &x, Real &y, Real &z);
void ConvVCarSph(const Real x, const Real y, const Real z, const Real vx, const Real vy, const Real vz, Real &vr, Real &vt, Real &vp);
void ConvVSphCar(const Real rad, const Real theta, const Real phi, const Real vr, const Real vt, const Real vp, Real &vx, Real &vy, Real &vz);



Real HstOutput(MeshBlock *pmb, int iout);

void Mesh::InitUserMeshData(ParameterInput *pin) 
{
  //read variables from the input file
  rhofloor = pin->GetOrAddReal("hydro", "dfloor", 1.e-8);
  //Get parameters for gravitatonal potential of central point mass

  gm = pin->GetOrAddReal("problem", "GM", 0.001); //potential of the star itself
  rm2 = pin->GetOrAddReal("problem","r0",1.0); //The orbital radius, or binary separation
  gm_com = pin->GetOrAddReal("problem","mass0",1.0); //The companion mass
  omegarot = pin->GetOrAddReal("problem","omegarot",1.0); //The orbital frequency
    tfloor = pin->GetOrAddReal("radiation", "tfloor", 0.01);

  //Get parameters of disk density (fed through outer radial boundary)

  den0 = pin->GetReal("problem","rho0");
  if(NON_BAROTROPIC_EOS)
  p0_over_r0 = pin->GetOrAddReal("problem","p0_over_r0",1.0); //midplane cs^2

  //get radiation-related parameters
    if(NR_RADIATION_ENABLED|| IM_RADIATION_ENABLED){
        //timeunit = pin->GetOrAddReal("radiation", "timeunit", 289977.36);  // 0.3 solar mass at 0.1 AU
        rhounit = pin->GetOrAddReal("radiation", "density_unit", 1.e-8);
        tunit = pin->GetOrAddReal("radiation", "T_unit", 1.0);
        lunit = pin->GetOrAddReal("radiation", "length_unit", 1.496e12);  // 0.1 AU
        mu = pin->GetOrAddReal("radiation", "molecular_weight", 1.);
        midprat = pin->GetOrAddReal("radiation","prat",0.0);
        midprat = midprat*p0_over_r0*p0_over_r0*p0_over_r0/den0;
        //printf("midplane r-g ratio: %e\n",midprat/3.0);
        tfloor = pin->GetOrAddReal("radiation", "tfloor", 0.01);
    }

/////////////////////////////////////////////////////////////// Enroll boundary functions

if(mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, Inflow_X1);
  if (NR_RADIATION_ENABLED|| IM_RADIATION_ENABLED) 
  EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, Inflow_rad_X1);
} //Can choose not to enroll radial inner boundary condition

if(mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, SteadyOuterX1); //feed in a disc
  if (NR_RADIATION_ENABLED|| IM_RADIATION_ENABLED) 
  EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, FixRadOuterX1new);
}
  EnrollUserExplicitSourceFunction(GravityPotential);

  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, HstOutput, "Mdot"); //measured accretion rate
  
//////////////////////////////////////////////////////////////initial condition table


  ini_profile.NewAthenaArray(in_line,6);
  FILE *fini;
  if ( (fini=fopen("./M50_MESA_25.txt","r"))==NULL )
  {
     printf("Open input file error MESA profile");
     return;
  }  

  for(int j=0; j<in_line; j++){
    for(int i=0; i<6; i++){
      fscanf(fini,"%lf",&(ini_profile(j,i)));

    }
  }

  fclose(fini);
  bd_data.NewAthenaArray(3,NGHOST); //boundary data



  x1length = mesh_size.nx1+1+2*NGHOST; // this is the total number of face points
  rloc.NewAthenaArray(x1length);
  Real x1ratio = mesh_size.x1rat;
  Real x1min = mesh_size.x1min;

  for(int i=NGHOST; i<x1length; ++i)
    rloc(i) = x1min * pow(x1ratio,i-NGHOST);

  for(int i=0; i<NGHOST; ++i)
    rloc(i) = x1min * pow(x1ratio,i-NGHOST);
  
  
///////////////////////////////// radiation /////////
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    //EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, Outflow_rad_X2);
  
    // the opacity table
    opacitytable.NewAthenaArray(212,46);
    planckopacity.NewAthenaArray(138,37);
    logttable.NewAthenaArray(212);
    logrhottable.NewAthenaArray(46);
    logttable_planck.NewAthenaArray(138);
    logrhottable_planck.NewAthenaArray(37);
    
    // read in the opacity table
    FILE *fkappa, *flogt, *flogrhot, *fplanck, *flogt_planck, *flogrhot_planck;
      
    if ( (fkappa=fopen("./aveopacity_combined.txt","r"))==NULL )
    {
      printf("Open input file error aveopacity_combined");
      return;
    }

    if ( (fplanck=fopen("./PlanckOpacity.txt","r"))==NULL )
    {
      printf("Open input file error PlanckOpacity");
      return;
    }

    if ( (flogt=fopen("./logT.txt","r"))==NULL )
    {
      printf("Open input file error logT");
      return;
    }

    if ( (flogrhot=fopen("./logRhoT.txt","r"))==NULL )
    {
      printf("Open input file error logRhoT");
      return;
    }

    if ( (flogt_planck=fopen("./logT_planck.txt","r"))==NULL )
    {
      printf("Open input file error logT_planck");
      return;
    }

    if ( (flogrhot_planck=fopen("./logRhoT_planck.txt","r"))==NULL )
    {
      printf("Open input file error logRhoT_planck");
      return;
    }

    for(int j=0; j<212; j++){
      for(int i=0; i<46; i++){
          fscanf(fkappa,"%lf",&(opacitytable(j,i)));
      }
    }

    for(int j=0; j<138; j++){
      for(int i=0; i<37; i++){
          fscanf(fplanck,"%lf",&(planckopacity(j,i)));
      }
     }


    for(int i=0; i<46; i++){
      fscanf(flogrhot,"%lf",&(logrhottable(i)));
    }

    for(int i=0; i<212; i++){
      fscanf(flogt,"%lf",&(logttable(i)));
    }

    for(int i=0; i<37; i++){
      fscanf(flogrhot_planck,"%lf",&(logrhottable_planck(i)));
    }

    for(int i=0; i<138; i++){
      fscanf(flogt_planck,"%lf",&(logttable_planck(i)));
    }

    fclose(fkappa);
    fclose(flogt);
    fclose(flogrhot);
    fclose(fplanck);
    fclose(flogt_planck);
    fclose(flogrhot_planck);
 
  }

if (adaptive) {
    EnrollUserRefinementCondition(RefinementCondition);
  }

  
//////end of opacity table///////
  return;
}


//======================================================================================
//! \fn void Mesh::TerminateUserMeshProperties(void)
//  \brief Clean up the Mesh properties
//======================================================================================
void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{

  ini_profile.DeleteAthenaArray();
  bd_data.DeleteAthenaArray();
  rloc.DeleteAthenaArray();

  // free memory
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){

    opacitytable.DeleteAthenaArray();
    logttable.DeleteAthenaArray();
    logrhottable.DeleteAthenaArray();
    planckopacity.DeleteAthenaArray();
    logttable_planck.DeleteAthenaArray();
    logrhottable_planck.DeleteAthenaArray();

  }


  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{

  //AllocateUserOutputVariables(n_user_var);
  
  // get bottom boundary condition

  Real rbottom = pcoord->x1v(is-1);
  if(rbottom < pmy_mesh->mesh_size.x1min){
    for(int i=1; i<=NGHOST; ++i){
      Real radius = pcoord->x1v(is-i);
      int lleft=0;

      int lright=1;
      while((radius > ini_profile(lright,0)) && (lright < in_line-1)){
        lright = lright+1;
      }
      if(lright - lleft > 1) lleft = lright -1;

      Real rho = ini_profile(lleft,2) + (radius - ini_profile(lleft,0)) *
                                (ini_profile(lright,2) - ini_profile(lleft,2))
                               /(ini_profile(lright,0) - ini_profile(lleft,0));
      Real tem = ini_profile(lleft,1) + (radius - ini_profile(lleft,0)) *
                                (ini_profile(lright,1) - ini_profile(lleft,1))
                               /(ini_profile(lright,0) - ini_profile(lleft,0));

      Real fr = ini_profile(lleft,5) + (radius - ini_profile(lleft,0)) *
                                (ini_profile(lright,5) - ini_profile(lleft,5))
                               /(ini_profile(lright,0) - ini_profile(lleft,0));

      bd_data(0,i-1) = rho;
      bd_data(1,i-1) = tem;
      bd_data(2,i-1) = fr;
    }

  }
 
  
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    
      pnrrad->EnrollOpacityFunction(StarOpacity);

  }else{

  }
  return;
}


void MeshBlock::UserWorkInLoop(void)
{
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    int il=is, iu=ie, jl=js, ju=je, kl=ks, ku=ke;
    il -= NGHOST;
    iu += NGHOST;
    if(ju>jl){
       jl -= NGHOST;
       ju += NGHOST;
    }
    if(ku>kl){
      kl -= NGHOST;
      ku += NGHOST;
    }
    Real gamma1 = peos->GetGamma() - 1.0;
    AthenaArray<Real> ir_cm;
    ir_cm.NewAthenaArray(pnrrad->n_fre_ang);
      
     for (int k=kl; k<=ku; ++k){
      for (int j=jl; j<=ju; ++j){
       for (int i=il; i<=iu; ++i){
         
          Real& vx=phydro->w(IVX,k,j,i);
          Real& vy=phydro->w(IVY,k,j,i);
          Real& vz=phydro->w(IVZ,k,j,i);
         
          Real& rho=phydro->w(IDN,k,j,i);
          Real& pgas=phydro->w(IEN,k,j,i);
         
          Real vel = sqrt(vx*vx+vy*vy+vz*vz);
//          if(vel > pnrrad->vmax * pnrrad->crat || pgas/rho > 1.e2){
//            printf("x1v: %e x2v: %e x3v: %e vel: %e tgas: %e\n",pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),vel,pgas/rho);
//          }

          if(vel > pnrrad->vmax * pnrrad->crat){
            Real ratio = pnrrad->vmax * pnrrad->crat / vel;
            vx *= ratio;
            vy *= ratio;
            vz *= ratio;
            
            phydro->u(IM1,k,j,i) = rho*vx;
            phydro->u(IM2,k,j,i) = rho*vy;
            phydro->u(IM3,k,j,i) = rho*vz;


            Real ke = 0.5 * rho * (vx*vx+vy*vy+vz*vz);
            
            Real pb=0.0;
            if(MAGNETIC_FIELDS_ENABLED){
               pb = 0.5*(SQR(pfield->bcc(IB1,k,j,i))+SQR(pfield->bcc(IB2,k,j,i))
                     +SQR(pfield->bcc(IB3,k,j,i)));
            }
            
            Real  eint = phydro->w(IEN,k,j,i)/gamma1;
            
            phydro->u(IEN,k,j,i) = eint + ke + pb;

          }
      }}}
      
      ir_cm.DeleteAthenaArray();
      
    }else{
      int il=is, iu=ie, jl=js, ju=je, kl=ks, ku=ke;
      il -= NGHOST;
      iu += NGHOST;
      if(ju>jl){
        jl -= NGHOST;
        ju += NGHOST;
      }
      if(ku>kl){
        kl -= NGHOST;
        ku += NGHOST;
      }
      //Real gamma1 = peos->GetGamma() - 1.0;
    
      for (int k=kl; k<=ku; ++k){
       for (int j=jl; j<=ju; ++j){
        for (int i=il; i<=iu; ++i){
         
          Real& vx=phydro->w(IVX,k,j,i);
          Real& vy=phydro->w(IVY,k,j,i);
          Real& vz=phydro->w(IVZ,k,j,i);
         
          Real& rho=phydro->w(IDN,k,j,i);
          
          Real tgas=phydro->w(IEN,k,j,i)/rho;
        }
       }
      }
    }
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief beam test
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  
  //Real gamma = peos->GetGamma();


  //initialize random number
  std::srand(gid);
 

  
  Real crat, prat;
  if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
    crat = pnrrad->crat;
    prat = pnrrad->prat;
  }else{
    crat = 9166.13;
    prat = 0.0;
  }
  
  Real amp = 0.0;
  
  
  int kl=ks, ku=ke;
  if(ku > kl){
    ku += NGHOST;
    kl -= NGHOST;
  }
  int jl=js, ju=je;
  if(ju > jl){
    ju += NGHOST;
    jl -= NGHOST;
  }
  int il = is-NGHOST, iu=ie+NGHOST;

  // First, find density and temperature at rmax
  int lleft=0;
  int lright=1;
  while((rmax > ini_profile(lright,0)) && (lright < in_line-1)){
     lright = lright+1;
  }
  if(lright - lleft > 1) lleft = lright -1;
  
//  Real rho_rmax = ini_profile(lleft,2) + (rmax - ini_profile(lleft,0)) *
//                              (ini_profile(lright,2) - ini_profile(lleft,2))
//                             /(ini_profile(lright,0) - ini_profile(lleft,0));
//  Real tem_rmax = ini_profile(lleft,1) + (rmax - ini_profile(lleft,0)) *
//                              (ini_profile(lright,1) - ini_profile(lleft,1))
//                             /(ini_profile(lright,0) - ini_profile(lleft,0));

  Real rho_rmax = ini_profile(in_line-1,2);
  Real tem_rmax = ini_profile(in_line-1,1);
  Real mass_rmax = ini_profile(in_line-1,3);

  Real grav_rmax = grav0*pow(lbottom/rmax,2.0)*(mass_rmax/massbottom);

  // Initialize the mass array



  
  // Initialize hydro variable
  for(int i=is; i<=ie; ++i) {
    Real &x1 = pcoord->x1v(i); 
    Real tem = std::max(tem_rmax, tfloor);
    Real rho = rho_rmax;   

    // get the position

    Real radflx = 0.0;


    if(x1 > rmax){
      tem = std::max(tem_rmax, tfloor);
      Real grav_local = grav_rmax * pow(x1/rmax,2.0);
      rho = rho_rmax * exp(-grav_local*(x1-rmax)/(tem));
      rho = std::max(rho,1.e-8);

    }else{
      int lleft=0;

      int lright=1;
      while((x1 > ini_profile(lright,0)) && (lright < in_line-1)){
         lright = lright+1;
      }
      if(lright - lleft > 1) lleft = lright -1;
      
      rho = ini_profile(lleft,2) + (x1 - ini_profile(lleft,0)) *
                                  (ini_profile(lright,2) - ini_profile(lleft,2))
                                 /(ini_profile(lright,0) - ini_profile(lleft,0));
      tem = ini_profile(lleft,1) + (x1 - ini_profile(lleft,0)) *
                                  (ini_profile(lright,1) - ini_profile(lleft,1))
                                 /(ini_profile(lright,0) - ini_profile(lleft,0));

      radflx = ini_profile(lleft,5) + (x1 - ini_profile(lleft,0)) *
                                  (ini_profile(lright,5) - ini_profile(lleft,5))
                                 /(ini_profile(lright,0) - ini_profile(lleft,0));

    }


    
    if(rho > 0.01 ) amp = 0.0;
    else amp = 0.0;

    
 //   rho *= (1.0 + amp * ((double)rand()/(double)RAND_MAX-0.5));
    
    
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        phydro->u(IDN,k,j,i) = rho * (1.0 + amp * ((double)rand()/(double)RAND_MAX-0.5));
        Real &x2 = pcoord->x2v(j); 
        Real &x3 = pcoord->x3v(k); 
        Real v1, v2, v3;
        VelProfile(x1, x2, x3, rho, v1, v2, v3);
        phydro->u(IM1,k,j,i) = 0.0;//rho*v1;
        phydro->u(IM2,k,j,i) = 0.0; //rho*v2;
        phydro->u(IM3,k,j,i) = 0.0; //rho*v3;
        if (NON_BAROTROPIC_EOS){

          phydro->u(IEN,k,j,i) = 1.5 * tem * rho + 0.5486219083605187 * tem * tem * tem * tem;
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM2,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i);
        }
        
        if(NR_RADIATION_ENABLED || IM_RADIATION_ENABLED){
          Real er = tem * tem * tem * tem;
          // geometric dialution
          
          for(int ifr=0; ifr<pnrrad->nfreq; ++ifr){
            Real coefa = 0.0, coefb = 0.0;
            for(int n=0; n<pnrrad->nang; ++n){
              // spherical polar coordinate
              Real &miuz = pnrrad->mu(0,k,j,i,n);
              Real &weight = pnrrad->wmu(n);
              if(miuz > 0.0){
                coefa += weight;
                coefb += (miuz * weight);
              }
            }
            
            for(int n=0; n<pnrrad->nang; ++n){
              Real &miuz = pnrrad->mu(0,k,j,i,n);
            
              if(miuz > 0.0){
                pnrrad->ir(k,j,i,ifr*pnrrad->nang+n) = 0.5 *
                                       (er/coefa + radflx/coefb);
              }else{
                pnrrad->ir(k,j,i,ifr*pnrrad->nang+n) = 0.5 *
                                       (er/coefa - radflx/coefb);
              
              }
            
            }
            
          }
        }// End Rad
 
      }// end j
    }// end k
  }// end i

  // Opacity will be set during initialization

  
  return;
}


static void VelProfile(const Real x1, const Real x2, const Real x3,
                                const Real den, Real &v1, Real &v2, Real &v3) 
{
  std::stringstream msg;
  Real xsf,ysf,zsf,xpf,ypf,zpf,rcylsf,rcyld,den0, rsphsf, thetasf, phisf;
  Real vrsphsf, vthetasf, vphisf, vxsf, vysf, vzsf;
  ConvSphCar(x1, x2, x3, xpf, ypf, zpf); //convert spherical x1x2x3 to xyz centered on the planet
  xsf=xpf;
  ysf=ypf-rm2; //star placed at y = r0
  zsf=zpf; 
  ConvCarSph(xsf, ysf, zsf, rsphsf, thetasf, phisf); //convert xyz centered on the star to spherical on star
  VelProfilesf(rsphsf, thetasf, phisf, den, vrsphsf, vthetasf, vphisf); //outputting vr vtheta vphi as spherical velocity
  ConvVSphCar(rsphsf, thetasf, phisf, vrsphsf, vthetasf, vphisf, vxsf, vysf, vzsf);
  ConvVCarSph(xpf, ypf, zpf, vxsf, vysf, vzsf, v1, v2, v3);
  return;
}




static void VelProfilesf(const Real x1, const Real x2, const Real x3,
               const Real den, Real &v1, Real &v2, Real &v3)
{
  std::stringstream msg;
  Real r = fabs(x1*sin(x2));
  Real z = fabs(x1*cos(x2)); //cylindrical coordinates
  Real vel = sqrt(gm_com/r); //the Keplerian velocity
    v1 = 0.0;
    v2 = 0.0;
    v3 = vel;
  if(omegarot!=0.0) v3-=omegarot*fabs(x1*sin(x2));  //subtract off rotation of the system
    Real fmax = 2.0;
    Real factor = 1.0;  //+ (fmax-1.0)*(1.0-time*fmax*omegarot);
    if (factor < 1.0) factor = 1.0;
    v3 = v3*factor;
  return;
}


void StarOpacity(MeshBlock *pmb, AthenaArray<Real> &prim)
{
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
      // electron scattering opacity
  Real kappas = 0.2 * (1.0 + 0.73);
  Real kappaa = 0.0;
  
  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
  for (int i=il; i<=iu; ++i) {
  for (int ifr=0; ifr<pnrrad->nfreq; ++ifr){
    Real rho  = prim(IDN,k,j,i);
    Real gast = std::max(prim(IEN,k,j,i)/rho,tfloor);
    Real kappa, kappa_planck;
    rossopacity(rho, gast, kappa, kappa_planck);

    if(kappa < kappas){
      if(gast < 1.0){
        kappaa = kappa;
        kappa = 0.0;
      }else{
        kappaa = 0.0;
      }
    }else{
      kappaa = kappa - kappas;
      kappa = kappas;
    }

    pnrrad->sigma_s(k,j,i,ifr) = kappa * rho * rhounit * lunit;
    pnrrad->sigma_a(k,j,i,ifr) = kappaa * rho * rhounit * lunit;
    pnrrad->sigma_p(k,j,i,ifr) = kappa_planck * rho * rhounit * lunit;
    pnrrad->sigma_pe(k,j,i,ifr) = pnrrad->sigma_p(k,j,i,ifr);

   // pnrrad->sigma_s(k,j,i,ifr) = kappa * rho * rhounit * lunit;
   // pnrrad->sigma_a(k,j,i,ifr) = kappaa * rho * rhounit * lunit;
   // pnrrad->sigma_ae(k,j,i,ifr) = pnrrad->sigma_a(k,j,i,ifr);
   // if(kappaa < kappa_planck)
   //   pnrrad->sigma_planck(k,j,i,ifr) = (kappa_planck-kappaa)*rho*rhounit*lunit;
   // else
   //   pnrrad->sigma_planck(k,j,i,ifr) = 0.0;
  }    
 }}}

}



void rossopacity(const Real rho, const Real tgas, Real &kappa, Real &kappa_planck)
{
  
    Real logt = log10(tgas * tunit);
    Real logrhot = log10(rho* rhounit) - 3.0* logt + 18.0;
    int nrhot1_planck = 0;
    int nrhot2_planck = 0;
    
    int nrhot1 = 0;
    int nrhot2 = 0;

    while((logrhot > logrhottable_planck(nrhot2_planck)) && (nrhot2_planck < 36)){
      nrhot1_planck = nrhot2_planck;
      nrhot2_planck++;
    }
    if(nrhot2_planck==36 && (logrhot > logrhottable_planck(nrhot2_planck)))
      nrhot1_planck=nrhot2_planck;

    while((logrhot > logrhottable(nrhot2)) && (nrhot2 < 45)){
      nrhot1 = nrhot2;
      nrhot2++;
    }
    if(nrhot2==45 && (logrhot > logrhottable(nrhot2)))
      nrhot1=nrhot2;
  
  /* The data point should between NrhoT1 and NrhoT2 */
    int nt1_planck = 0;
    int nt2_planck = 0;
    int nt1 = 0;
    int nt2 = 0;
    while((logt > logttable_planck(nt2_planck)) && (nt2_planck < 137)){
      nt1_planck = nt2_planck;
      nt2_planck++;
    }
    if(nt2_planck==137 && (logt > logttable_planck(nt2_planck)))
      nt1_planck=nt2_planck;

    while((logt > logttable(nt2)) && (nt2 < 211)){
      nt1 = nt2;
      nt2++;
    }
    if(nt2==211 && (logt > logttable(nt2)))
      nt1=nt2;

  

    Real kappa_t1_rho1=opacitytable(nt1,nrhot1);
    Real kappa_t1_rho2=opacitytable(nt1,nrhot2);
    Real kappa_t2_rho1=opacitytable(nt2,nrhot1);
    Real kappa_t2_rho2=opacitytable(nt2,nrhot2);

    Real planck_t1_rho1=planckopacity(nt1_planck,nrhot1_planck);
    Real planck_t1_rho2=planckopacity(nt1_planck,nrhot2_planck);
    Real planck_t2_rho1=planckopacity(nt2_planck,nrhot1_planck);
    Real planck_t2_rho2=planckopacity(nt2_planck,nrhot2_planck);


    // in the case the temperature is out of range
    // the planck opacity should be smaller by the 
    // ratio T^-3.5
    if(nt2_planck == 137 && (logt > logttable_planck(nt2_planck))){
       Real scaling = pow(10.0, -3.5*(logt - logttable_planck(137)));
       planck_t1_rho1 *= scaling;
       planck_t1_rho2 *= scaling;
       planck_t2_rho1 *= scaling;
       planck_t2_rho2 *= scaling;
    }


    Real rho_1 = logrhottable(nrhot1);
    Real rho_2 = logrhottable(nrhot2);
    Real t_1 = logttable(nt1);
    Real t_2 = logttable(nt2);

    
    if(nrhot1 == nrhot2){
      if(nt1 == nt2){
        kappa = kappa_t1_rho1;
      }else{
        kappa = kappa_t1_rho1 + (kappa_t2_rho1 - kappa_t1_rho1) *
                                (logt - t_1)/(t_2 - t_1);
      }/* end same T*/
    }else{
      if(nt1 == nt2){
        kappa = kappa_t1_rho1 + (kappa_t1_rho2 - kappa_t1_rho1) *
                                (logrhot - rho_1)/(rho_2 - rho_1);
      }else{
        kappa = kappa_t1_rho1 * (t_2 - logt) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
              + kappa_t2_rho1 * (logt - t_1) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
              + kappa_t1_rho2 * (t_2 - logt) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
              + kappa_t2_rho2 * (logt - t_1) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1));
      }
    }/* end same rhoT */

    rho_1 = logrhottable_planck(nrhot1_planck);
    rho_2 = logrhottable_planck(nrhot2_planck);
    t_1 = logttable_planck(nt1_planck);
    t_2 = logttable_planck(nt2_planck);
 
  /* Now do the same thing for Planck mean opacity */
    if(nrhot1_planck == nrhot2_planck){
      if(nt1_planck == nt2_planck){
        kappa_planck = planck_t1_rho1;
      }else{
        kappa_planck = planck_t1_rho1 + (planck_t2_rho1 - planck_t1_rho1) *
                                (logt - t_1)/(t_2 - t_1);
      }/* end same T*/
    }else{
      if(nt1_planck == nt2_planck){
        kappa_planck = planck_t1_rho1 + (planck_t1_rho2 - planck_t1_rho1) *
                                (logrhot - rho_1)/(rho_2 - rho_1);

      }else{        
        kappa_planck = planck_t1_rho1 * (t_2 - logt) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
                     + planck_t2_rho1 * (logt - t_1) * (rho_2 - logrhot)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
                     + planck_t1_rho2 * (t_2 - logt) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1))
                     + planck_t2_rho2 * (logt - t_1) * (logrhot - rho_1)/
                                ((t_2 - t_1) * (rho_2 - rho_1));
      }
    }/* end same rhoT */

    return;

}

// This function sets boundary condition for primitive variables

void Inflow_X1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{

  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {

        Real rho = bd_data(0,i-1);
        Real tem = bd_data(1,i-1);
//        Real tem = tbd;
        prim(IDN,k,j,is-i) = bd_data(0,i-1);
        prim(IVX,k,j,is-i) = -prim(IVX,k,j,is+i-1);
        prim(IVY,k,j,is-i) = -prim(IVY,k,j,is+i-1);
        prim(IVZ,k,j,is-i) = -prim(IVZ,k,j,is+i-1);
        prim(IEN,k,j,is-i) = prim(IDN,k,j,is-i)*tem  + 0.5486219083605187 * tem * tem * tem * tem / 3.0;
        
      }
    }
  }
   // set magnetic field in inlet ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for(int k=ks; k<=ke; ++k){
    for(int j=js; j<=je; ++j){
      for(int i=1; i<=ngh; ++i){
        b.x1f(k,j,is-i) = b.x1f(k,j,is);
      }
    }}

    for(int k=ks; k<=ke; ++k){
    for(int j=js; j<=je+1; ++j){
      for(int i=1; i<=ngh; ++i){
        b.x2f(k,j,is-i) = b.x2f(k,j,is);
      }
    }}

    for(int k=ks; k<=ke+1; ++k){
    for(int j=js; j<=je; ++j){
      for(int i=1; i<=ngh; ++i){
        b.x3f(k,j,is-i) = b.x3f(k,j,is);
      }
    }}
    
  }


  return;
}

void Outflow_X2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  //initialize random number

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
          Real &x1g = pco->x1v(ie+i);
          Real &x1 = pco->x1v(ie+i-1);
          if(prim(IVX,k,j,ie) < 0.0){
            prim(IDN,k,j,ie+i) = prim(IDN,k,j,ie);
            prim(IVX,k,j,ie+i) = 0.0;
          }else{
            prim(IDN,k,j,ie+i) = prim(IDN,k,j,ie+i-1) * x1*x1/(x1g*x1g);
            prim(IVX,k,j,ie+i) = prim(IVX,k,j,ie);
          }
          prim(IVY,k,j,ie+i) = prim(IVY,k,j,ie);
          prim(IVZ,k,j,ie+i) = prim(IVZ,k,j,ie);
          prim(IEN,k,j,ie+i) = prim(IPR,k,j,ie);
        
      }
    }
  }
  

  return;
}



void SteadyOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real x1 = pco->x1v(ie+i);
          Real x2 = pco->x2v(j);
          Real x3 = pco->x3v(k);
          Real z = fabs(x1*cos(x2));
          Real zfactor = (1-SQR(z)/8/p0_over_r0/(1 + midprat/3)*omegarot*omegarot);
          Real den = den0*zfactor*zfactor*zfactor;
    
          if (den < rhofloor) {den = rhofloor;}
         
          prim(IDN,k,j,ie+i) =den ;
          Real v1, v2, v3;
          VelProfile(x1, x2, x3, prim(IDN,k,j,ie+i), v1, v2, v3);
          prim(IM1,k,j,ie+i) = v1;
          prim(IM2,k,j,ie+i) = v2;
          prim(IM3,k,j,ie+i) = v3;
          
          if (NON_BAROTROPIC_EOS) {
            Real tem = p0_over_r0*zfactor;
            if (tem < tfloor) {tem = tfloor;}
            //if (j==je)
            //printf("midplane temperature: %e\n",tem);
            prim(IEN,k,j,ie+i) = prim(IDN,k,j,ie+i)*tem  + 0.5486219083605187 * tem * tem * tem * tem / 3.0;
          //Real press = PoverR(x1, x2, x3)*prim(IDN,k,j,ie+i)/3;
          //prim(IEN,k,j,ie+i) = (1.5*press + 3*press*p0_over_r0)/(1+ p0_over_r0); for general eos
          }
        }
      }
    }
  }
}




void FixRadOuterX1new(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad, 
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh){
    AthenaArray<Real> ir_cm;
    Real *ir_lab;
    ir_cm.NewAthenaArray(pnrrad->n_fre_ang);
   
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          

          Real &x1 = pco->x1v(ie+i);
          Real &x2 = pco->x2v(j);
          Real &x3 = pco->x3v(k);
          Real z = fabs(x1*cos(x2));
          Real zfactor = (1-SQR(z)/8/p0_over_r0/(1 + midprat/3)*omegarot*omegarot);
          Real tem = p0_over_r0*zfactor;
          if (tem < tfloor) {tem = tfloor;}
          Real er = tem * tem * tem * tem;


        for(int ifr=0; ifr<pnrrad->nfreq; ++ifr){
          Real coefa = 0.0, coefb = 0.0;
          for(int n=0; n<pnrrad->nang; ++n){
            // spherical polar coordinate
            Real &miuz = pnrrad->mu(0,k,j,ie+i,n);
            Real &weight = pnrrad->wmu(n);
            if(miuz > 0.0){
              coefa += weight;
              coefb += (miuz * weight);
            }
          }

          for(int n=0; n<pnrrad->nang; ++n){ 
            Real &miuz = pnrrad->mu(0,k,j,ie+i,n);
            if(miuz > 0.0)
              ir_cm(n) = 0.5 * (er/coefa);
            else
              ir_cm(n) = 0.5 * (er/coefa);
          }
          Real *mux = &(pnrrad->mu(0,k,j,ie+i,0));
          Real *muy = &(pnrrad->mu(1,k,j,ie+i,0));
          Real *muz = &(pnrrad->mu(2,k,j,ie+i,0));

          ir_lab = &(ir(k,j,ie+i,0));
          
            
          for(int n=0; n<pnrrad->nang; ++n){
            ir_lab[n] = ir_cm(n);
          }  
          
          pnrrad->pradintegrator->ComToLab(w(IVX,k,j,ie+i),w(IVY,k,j,ie+i),w(IVZ,k,j,ie+i),mux,muy,muz,ir_cm,ir_lab);
         
        }

}//i
}//j
}//k

ir_cm.DeleteAthenaArray();
return;
}



void Outflow_rad_X2(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad, 
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  
  //vacuum boundary condition at top   
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {

        for(int ifr=0; ifr<pnrrad->nfreq; ++ifr){
          for(int n=0; n<pnrrad->nang; ++n){
            Real miuz = pnrrad->mu(0,k,j,ie+i,ifr*pnrrad->nang+n);
            if(miuz > 0.0){
              ir(k,j,ie+i,ifr*pnrrad->nang+n)
                            = ir(k,j,ie+i-1,ifr*pnrrad->nang+n);
            }else{
              ir(k,j,ie+i,ifr*pnrrad->nang+n) = 0.0;
            }
         
            
          }
        }
      }
    }
  }
  

  return;
}


//inpose a fix density, temperature, luminosity
void Inflow_rad_X1(MeshBlock *pmb, Coordinates *pco, NRRadiation *pnrrad, 
     const AthenaArray<Real> &w, FaceField &b, AthenaArray<Real> &ir, 
      Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
 
  AthenaArray<Real> ir_cm;
  Real *ir_lab;
  ir_cm.NewAthenaArray(pnrrad->n_fre_ang);
 
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {

        Real &x1 = pco->x1v(is-i);    
        Real radflx = bd_data(2,i-1);

        Real tem = bd_data(1,i-1);
//        Real tem = tbd;
        Real er = tem * tem * tem * tem;

        
        for(int ifr=0; ifr<pnrrad->nfreq; ++ifr){
          Real coefa = 0.0, coefb = 0.0;
          for(int n=0; n<pnrrad->nang; ++n){
            // spherical polar coordinate
            Real &miuz = pnrrad->mu(0,k,j,is-i,n);
            Real &weight = pnrrad->wmu(n);
            if(miuz > 0.0){
              coefa += weight;
              coefb += (miuz * weight);
            }
          }

          for(int n=0; n<pnrrad->nang; ++n){ 
            Real &miuz = pnrrad->mu(0,k,j,is-i,n);
            if(miuz > 0.0)
              ir_cm(n) = 0.5 * (er/coefa + radflx/coefb);
            else
              ir_cm(n) = 0.5 * (er/coefa - radflx/coefb);
          }
          Real *mux = &(pnrrad->mu(0,k,j,is-i,0));
          Real *muy = &(pnrrad->mu(1,k,j,is-i,0));
          Real *muz = &(pnrrad->mu(2,k,j,is-i,0));

          ir_lab = &(ir(k,j,is-i,0));
    
          pnrrad->pradintegrator->ComToLab(w(IVX,k,j,is-i),w(IVY,k,j,is-i),w(IVZ,k,j,is-i),mux,muy,muz,ir_cm,ir_lab);
         
        }
          
      }//i
    }//j
  }//k
  
  ir_cm.DeleteAthenaArray();
  return;
}




//gravitational potential for the tidal term, not including the RSG

Real grav_pot(const Real radius, const Real theta, const Real phi)
{
  // the companion is located at \theta=90, phi=pi/2, r=rm2
  //x=0, y=rm2, z=0
  // current point r\sin\theta \cosphi, r\sin\theta\sin phi, r\sin\theta
  Real dist_r2=sqrt(radius*radius+rm2*rm2-2.0*radius*rm2*sin(theta)*sin(phi));

  Real potphi=-gm_com/sqrt(dist_r2*dist_r2)-0.5*omegarot*omegarot*radius*radius  //midplane projection is radius*sintheta
        *sin(theta)*sin(theta)+gm_com*radius*sin(theta)*sin(phi)/(rm2*rm2);
  return potphi;
}


//Gravitaional acceration takes the form
// GM = grav0(r/r_0)^-2
// The potential is 
//phi = -grav0*r0 r^-1
// grav=-\partial \phi/\partial r

void GravityPotential(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons, AthenaArray<Real> &cons_scalar)
{

  Coordinates *pco = pmb->pcoord;
  AthenaArray<Real> &x1flux=pmb->phydro->flux[X1DIR];
  AthenaArray<Real> &x2flux=pmb->phydro->flux[X2DIR];
  AthenaArray<Real> &x3flux=pmb->phydro->flux[X3DIR];

  for(int k=pmb->ks; k<=pmb->ke; ++k){
    for(int j=pmb->js; j<=pmb->je; ++j){
      for(int i=pmb->is; i<=pmb->ie; ++i){
        Real rho = prim(IDN,k,j,i);
        Real rcen = pmb->pcoord->x1v(i);
        Real rleft = pmb->pcoord->x1f(i);
        Real rright = pmb->pcoord->x1f(i+1);
      
        // first, add GM/r^2 term due to RGB

        //Real src = - dt * rho *gm /(rcen * rcen);

        //No need now, since GM already takes care of it
        //cons(IM1,k,j,i) += src;
        //cons(IEN,k,j,i) -= 0.5*dt*(x1flux(IDN,k,j,i) + x1flux(IDN,k,j,i+1))* gm /(rcen * rcen);

        //pmb->user_out_var(0,k,j,i) = src/dt;
        //pmb->user_out_var(1,k,j,i) = -0.5*(x1flux(IDN,k,j,i) + x1flux(IDN,k,j,i+1)) 
                           //* gm /(rcen * rcen);

         // now add the tidal term
        Real src = 0.0;

        Real thetacen = pmb->pcoord->x2v(j);
        Real thetaleft = pmb->pcoord->x2f(j);
        Real thetaright = pmb->pcoord->x2f(j+1);

        Real phicen = pmb->pcoord->x3v(k);
        Real phileft = pmb->pcoord->x3f(k);
        Real phiright = pmb->pcoord->x3f(k+1);

        Real vol=pmb->pcoord->GetCellVolume(k,j,i);
        Real phic = grav_pot(rcen,thetacen,phicen);

        // radial direction

        Real phil = grav_pot(rleft,thetacen,phicen);
        Real phir = grav_pot(rright,thetacen,phicen);

        Real areal=pmb->pcoord->GetFace1Area(k,j,i);
        Real arear=pmb->pcoord->GetFace1Area(k,j,i+1);

        src = - dt * rho * (phir - phil)/pmb->pcoord->dx1f(i);
        cons(IM1,k,j,i) += src;
        Real phidivrhov = (arear*x1flux(IDN,k,j,i+1) -
                           areal*x1flux(IDN,k,j,i))*phic/vol;
        Real divrhovphi = (arear*x1flux(IDN,k,j,i+1)*phir -
                           areal*x1flux(IDN,k,j,i)*phil)/vol;
        cons(IEN,k,j,i) += (dt*(phidivrhov - divrhovphi));

        //theta direction

        phil = grav_pot(rcen,thetaleft,phicen);
        phir = grav_pot(rcen,thetaright,phicen);

        areal=0.5*(rright*rright-rleft*rleft)*fabs(sin(thetaleft))*
                   pmb->pcoord->dx3f(k);
        arear=0.5*(rright*rright-rleft*rleft)*fabs(sin(thetaright))*
                   pmb->pcoord->dx3f(k);

        src = - dt * rho * (phir - phil)/(rcen*pmb->pcoord->dx2f(j));
        cons(IM2,k,j,i) += src;
        phidivrhov = (arear*x2flux(IDN,k,j+1,i) -
                           areal*x2flux(IDN,k,j,i))*phic/vol;
        divrhovphi = (arear*x2flux(IDN,k,j+1,i)*phir -
                           areal*x2flux(IDN,k,j,i)*phil)/vol;
        cons(IEN,k,j,i) += (dt*(phidivrhov - divrhovphi));

        //phi direction

        phil = grav_pot(rcen,thetacen,phileft);
        phir = grav_pot(rcen,thetacen,phiright);

        areal=0.5*(rright*rright-rleft*rleft)*pmb->pcoord->dx2f(j);
        arear=areal;

        src = - dt * rho * (phir - phil)/(rcen*fabs(sin(thetacen))*
                                pmb->pcoord->dx3f(k));
        cons(IM3,k,j,i) += src;
        phidivrhov = (arear*x3flux(IDN,k+1,j,i) -
                           areal*x3flux(IDN,k,j,i))*phic/vol;
        divrhovphi = (arear*x3flux(IDN,k+1,j,i)*phir -
                           areal*x3flux(IDN,k,j,i)*phil)/vol;
        cons(IEN,k,j,i) += (dt*(phidivrhov - divrhovphi));


        // Add the coriolis force
       //dM/dt=-2\rho \Omega_0\times V
       // Omega_0=(\Omega_0\cos\theta,-\Omega_0\sin\theta,0)
       // because we use semi-implicit method, we need velocity
       // from conservative quantities
        rho = cons(IDN,k,j,i);
        Real vr=cons(IVX,k,j,i)/rho;
        Real vtheta=cons(IVY,k,j,i)/rho;
        Real vphi=cons(IVZ,k,j,i)/rho;
        Real dtomega = dt*omegarot;
        Real sintheta=sin(thetacen);
        Real costheta=cos(thetacen);


        Real vphinew = -2.0 * sintheta*vr - 2.0*costheta*vtheta-(dtomega-1.0/dtomega)*vphi;
        vphinew /= (dtomega+1.0/dtomega);

        Real vrnew = dtomega * sintheta*vphinew + vr + dtomega*sintheta*vphi;
        Real vthetanew = dtomega * costheta*vphinew + vtheta + dtomega*costheta*vphi;

        cons(IM1,k,j,i) = vrnew * rho;

        cons(IM2,k,j,i) = vthetanew * rho;

        cons(IM3,k,j,i) = vphinew * rho;

        
      }
    }
  }

}



//Define outer boundary condition velocity




void ConvCarSph(const Real x, const Real y, const Real z, Real &rad, Real &theta, Real &phi){
  rad=sqrt(x*x+y*y+z*z);
  theta=acos(z/rad);
  phi=atan2(y,x);
  return;
}

void ConvSphCar(const Real rad, const Real theta, const Real phi, Real &x, Real &y, Real &z){
  x=rad*sin(theta)*cos(phi);
  y=rad*sin(theta)*sin(phi);
  z=rad*cos(theta);
  return;
}

void ConvVCarSph(const Real x, const Real y, const Real z, const Real vx, const Real vy, const Real vz, Real &vr, Real &vt, Real &vp){
  Real rads=sqrt(x*x+y*y+z*z);
  Real radc=sqrt(x*x+y*y);
  vr=vx*x/rads+vy*y/rads+vz*z/rads;
  vt=((x*vx+y*vy)*z-radc*radc*vz)/rads/radc;
  vp=vy*x/radc-vx*y/radc;
  return;
}

void ConvVSphCar(const Real rad, const Real theta, const Real phi, const Real vr, const Real vt, const Real vp, Real &vx, Real &vy, Real &vz){
  vx=vr*sin(theta)*cos(phi)+vt*cos(theta)*cos(phi)-vp*sin(phi);
  vy=vr*sin(theta)*sin(phi)+vt*cos(theta)*sin(phi)+vp*cos(phi);
  vz=vr*cos(theta)-vt*sin(theta);
  return;
}


//define boundary condition density



// refinement condition: check the maximum pressure gradient
int RefinementCondition(MeshBlock *pmb) {
  Real X1 = pmb->pcoord->x1v(pmb->is);
  Real X2 = pmb->pcoord->x2v(pmb->je);
  if ((X1 < 100))
    return 1;
  else
    return 0;
}




Real HstOutput(MeshBlock *pmb, int iout) {
  Real A = 0.;
  const Real r_sink = 100.0;
  Real n_cell_acc2 = 0;
  for (int i = pmb->is; i <= pmb->ie; ++i) {
  Real r = pmb->pcoord->x1v(i);
  if (r<2*r_sink && r>r_sink){
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
        //if (iout==0) { // accretion rate, then both Mdot and rho_center are displaying accretion rate
        Real dV = pmb->pcoord->GetCellVolume(k,j,i);
        A -= 4*PI*r*r*pmb->phydro->u(IM1,k,j,i)*dV;
        //n_cell_acc2 = 1; 
         //   }
          
        }
      }
    }
  }
  //A = A/n_cell_acc2;
  return A;
}

