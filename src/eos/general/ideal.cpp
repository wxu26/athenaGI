//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file ideal.cpp
//! \brief implements ideal EOS in general EOS framework, mostly for debuging
//======================================================================================

// C headers

// C++ headers

// Athena++ headers
#include "../eos.hpp"

namespace {
const Real rho_unit = 1.e-7;
const Real cs_unit = 3714005.4758361643;
const Real mu = 0.60276703256504693;
}

//takes in temperature and density and outputs pressure
Real pressureEq(Real temp, Real rho) {
  
  Real a=7.56*std::pow(10.0,-15.0);
  //Real mu=4.0/3.0;
  Real mProton=1.6726*std::pow(10.0,-24.0);
  Real kB=1.3807*std::pow(10.0,-16.0);
  return ((1/3.)*a*std::pow(temp,4.0))+rho*kB*temp/(mu*mProton);
}

//takes in temperature and density and outputs internal energy
Real energyEq(Real temp, Real rho) {

  Real a=7.56*std::pow(10.0,-15.0);
  //Real mu=4.0/3.0;
  Real mProton=1.6726*std::pow(10.0,-24.0);
  Real kB=1.3807*std::pow(10.0,-16.0);
  return (a*std::pow(temp,4.0))+1.5*rho*kB*temp/(mu*mProton);
}

//Solves the cubic x^4+4Bx-A^2=0 to get a root of our repressed cubic, formula off of wolfram
Real findRootCubic(Real A, Real B) {
  Real z3=std::pow(81.0*std::pow(A,4.0)+768.00*std::pow(B,3.0),.5)+9.0*A*A;
  Real numerator=std::pow(2.0,1/3.)*std::pow(z3, 2.0/3.0)-8.0*std::pow(3.0,1.0/3.0)*B;
  Real denominator= std::pow(6.0, 2.0/3.0) * std::pow(z3,1.0/3.0);
  return numerator/denominator;
}
//takes in pressure and density and solves the quartic analytically to get you temperature
Real calcTemperaturePressure(Real rho, Real pres) {
  Real a,mProton,kB,A,B,temp,z1,z2,y;
  mProton=1.6726*std::pow(10.0,-24.0);
  kB=1.3807*std::pow(10.0,-16.0);
  //mu=4.0/3.0;
  a=7.56*std::pow(10.0,-15.0);
  A=3.0*kB*rho/(a*mu*mProton);
  B=3.0*pres/a;
  y=findRootCubic(A,B);
  temp=std::pow(y,.5)*(std::pow(2.0*A/std::pow(y*y*y,.5)-1,.5)-1)/2.0;
  return temp;
}

//takes in energy and density and solves the quartic analytically to get you temeprature
Real calcTemperatureEnergy(Real rho, Real energy) {
  Real a,mProton,kB,A,B,temp,z1,z2,y;
  mProton=1.6726*std::pow(10.0,-24.0);
  kB=1.3807*std::pow(10.0,-16.0);
  //mu=4.0/3.0;
  a=7.56*std::pow(10.0,-15.0);
  A=3.0*kB*rho/(2.0*a*mu*mProton);
  B=energy/(a);
  y=findRootCubic(A,B);
  temp=std::pow(y,.5)*(std::pow(2.0*A/std::pow(y*y*y,.5)-1,.5)-1)/2.0; 
  return temp;
}

//takes in temperature pressure and density and returns gamma
Real calcGamma(Real rho, Real pres) {
  Real gasPres, a,  mProton, kB, beta, temp;
  mProton=1.6726*std::pow(10.0,-24.0);
  kB=1.3807*std::pow(10.0,-16.0);
  temp=calcTemperaturePressure(rho, pres);
  //mu=4.0/3.0;
  gasPres=rho*kB*temp/(mu*mProton);
  beta=gasPres/pres;
  return (32.0-24.0*beta-3.0*beta*beta)/(24.0-21.0*beta);
}
//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::PresFromRhoEg(Real rho, Real egas)
//! \brief Return gas pressure
Real EquationOfState::PresFromRhoEg(Real rho, Real egas) {
  Real temperature=calcTemperatureEnergy(rho*rho_unit, egas*rho_unit*cs_unit*cs_unit);
  return pressureEq(temperature,rho*rho_unit)/(rho_unit*cs_unit*cs_unit);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::EgasFromRhoP(Real rho, Real pres)
//! \brief Return internal energy density
Real EquationOfState::EgasFromRhoP(Real rho, Real pres) {
  Real temperature=calcTemperaturePressure(rho*rho_unit, pres*rho_unit*cs_unit*cs_unit);
  return energyEq(temperature, rho*rho_unit)/(rho_unit*cs_unit*cs_unit);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::AsqFromRhoP(Real rho, Real pres)
//! \brief Return adiabatic sound speed squared
Real EquationOfState::AsqFromRhoP(Real rho, Real pres) {
  Real gamma1=calcGamma(rho*rho_unit,pres*rho_unit*cs_unit*cs_unit);
  return gamma1 * pres / rho;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::InitEosConstants(ParameterInput* pin)
//! \brief Initialize constants for EOS
void EquationOfState::InitEosConstants(ParameterInput *pin) {
  return;
}

