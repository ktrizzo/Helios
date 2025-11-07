/** \file "EnergyBalanceModel_kokkos.cpp" Energy balance model plugin declarations (Kokkos kernels).

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include <Kokkos_Core.hpp>
#include "EnergyBalanceModel.h"

using namespace helios;
using namespace std;

// Device function for evaluating energy balance equation
KOKKOS_INLINE_FUNCTION
float evaluateEnergyBalance( float T, float R, float Qother, float eps, float Ta, float ea, float pressure, float gH, float gS, uint Nsides, float stomatal_sidedness, float heatcapacity, float surfacehumidity, float dt, float Tprev ){

    //Outgoing emission flux
    float Rout = float(Nsides)*eps*5.67e-8F*T*T*T*T;

    //Sensible heat flux
    float cp = 29.25f; //Molar specific heat of air. Units: J/mol
    float QH = cp*gH*(T-Ta); // (see Campbell and Norman Eq. 6.8)

    //Latent heat flux
    float es = 611.f*Kokkos::exp(17.502f*(T-273.f)/((T-273.f)+240.97f)); // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in Kelvin, and result is in Pascals
    float gM = 1.08f*gH*gS*(stomatal_sidedness/(1.08f*gH+gS*stomatal_sidedness) + (1.f-stomatal_sidedness)/(1.08f*gH+gS*(1.f-stomatal_sidedness)));
    if( gH==0 && gS==0 ){//if somehow both go to zero, can get NaN
        gM = 0;
    }
    float lambda = 44000.f; //Latent heat of vaporization for water. Units: J/mol
    float QL = gM*lambda*(es-ea*surfacehumidity)/pressure;

    //Storage heat flux
    float storage = 0.f;
    if (dt>0){
        storage=heatcapacity*(T-Tprev)/dt;
    }

    //Residual
    return R-Rout-QH-QL-Qother-storage;

}

void EnergyBalanceModel::run(){
    run( context->getAllUUIDs() );
}

void EnergyBalanceModel::run( float dt ){
    run( context->getAllUUIDs(), dt );
}

void EnergyBalanceModel::run( const std::vector<uint> &UUIDs ){
    run( UUIDs, 0.f);
}


void EnergyBalanceModel::run( const std::vector<uint> &UUIDs, float dt ){

    if( message_flag ){
        std::cout << "Running energy balance model..." << std::flush;
    }

    // Check that some primitives exist in the context

    uint Nprimitives = UUIDs.size();

    if( Nprimitives==0 ){
        std::cerr << "WARNING (EnergyBalanceModel::run): No primitives have been added to the context.  There is nothing to simulate. Exiting..." << std::endl;
        return;
    }

    //---- Sum up to get total absorbed radiation across all bands ----//

    // Look through all flux primitive data in the context and sum them up in vector Rn.  Each element of Rn corresponds to a primitive.

    if( radiation_bands.size()==0 ){
        helios_runtime_error("ERROR (EnergyBalanceModel::run): No radiation bands were found.");
    }

    std::vector<float> Rn;
    Rn.resize(Nprimitives,0);

    std::vector<float> emissivity;
    emissivity.resize(Nprimitives);
    for( size_t u=0; u<Nprimitives; u++ ){
        emissivity.at(u) = 1.f;
    }

    for( int b=0; b<radiation_bands.size(); b++ ){
        for( size_t u=0; u<Nprimitives; u++ ){
            size_t p = UUIDs.at(u);

            char str[50];
            sprintf(str,"radiation_flux_%s",radiation_bands.at(b).c_str());
            if( !context->doesPrimitiveDataExist(p,str) ) {
                helios_runtime_error("ERROR (EnergyBalanceModel::run): No radiation was found in the context for band " + std::string(radiation_bands.at(b)) + ". Did you run the radiation model for this band?");
            }else if( context->getPrimitiveDataType(p,str)!=HELIOS_TYPE_FLOAT ){
                helios_runtime_error("ERROR (EnergyBalanceModel::run): Radiation primitive data for band " + std::string(radiation_bands.at(b)) + " does not have the correct type of ''float'");
            }
            float R;
            context->getPrimitiveData(p,str,R);
            Rn.at(u) += R;

            sprintf(str,"emissivity_%s",radiation_bands.at(b).c_str());
            if( context->doesPrimitiveDataExist(p,str) && context->getPrimitiveDataType(p,str)==HELIOS_TYPE_FLOAT ){
                context->getPrimitiveData(p,str,emissivity.at(u));
            }

        }
    }

    //---- Set up temperature solution ----//

    // Allocate host arrays
    std::vector<float> To_host(Nprimitives);
    std::vector<float> R_host(Nprimitives);
    std::vector<float> Qother_host(Nprimitives);
    std::vector<float> eps_host(Nprimitives);
    std::vector<float> Ta_host(Nprimitives);
    std::vector<float> ea_host(Nprimitives);
    std::vector<float> pressure_host(Nprimitives);
    std::vector<float> gH_host(Nprimitives);
    std::vector<float> gS_host(Nprimitives);
    std::vector<uint> Nsides_host(Nprimitives);
    std::vector<float> stomatal_sidedness_host(Nprimitives);
    std::vector<float> heatcapacity_host(Nprimitives);
    std::vector<float> surfacehumidity_host(Nprimitives);

    bool calculated_blconductance_used = false;
    bool primitive_length_used = false;

    for( uint u=0; u<Nprimitives; u++ ){
        size_t p = UUIDs.at(u);

        //Initial guess for surface temperature
        if( context->doesPrimitiveDataExist(p,"temperature") && context->getPrimitiveDataType(p,"temperature")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"temperature",To_host[u]);
        }else{
            To_host[u] = temperature_default;
        }
        if( To_host[u]==0 ){//can't have To equal to 0
            To_host[u] = 300;
        }

        //Air temperature
        if( context->doesPrimitiveDataExist(p,"air_temperature") && context->getPrimitiveDataType(p,"air_temperature")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"air_temperature",Ta_host[u]);
            if( message_flag && Ta_host[u]<250.f ){
              std::cout << "WARNING (EnergyBalanceModel::run): Value of " << Ta_host[u] << " given in 'air_temperature' primitive data is very small. Values should be given in units of Kelvin. Assuming default value of " << air_temperature_default << std::endl;
              Ta_host[u] = air_temperature_default;
            }
        }else{
            Ta_host[u] = air_temperature_default;
        }

        //Air relative humidity
        float hr;
        if( context->doesPrimitiveDataExist(p,"air_humidity") && context->getPrimitiveDataType(p,"air_humidity")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"air_humidity",hr);
            if( hr>1.f ){
                if( message_flag ){
                    std::cout << "WARNING (EnergyBalanceModel::run): Value of " << hr << " given in 'air_humidity' primitive data is large than 1. Values should be given as fractional values between 0 and 1. Assuming default value of " << air_humidity_default << std::endl;
                }
                hr = air_humidity_default;
            }else if( hr<0.f ){
                if( message_flag ) {
                    std::cout << "WARNING (EnergyBalanceModel::run): Value of " << hr << " given in 'air_humidity' primitive data is less than 0. Values should be given as fractional values between 0 and 1. Assuming default value of " << air_humidity_default << std::endl;
                }
                hr = air_humidity_default;
            }
        }else{
            hr = air_humidity_default;
        }

        //Air vapor pressure
        float esat = 611.f*exp(17.502f*(Ta_host[u]-273.f)/((Ta_host[u]-273.f)+240.97f)); // This is Clausius-Clapeyron equation (See Campbell and Norman pp. 41 Eq. 3.8).  Note that temperature must be in degC, and result is in Pascals
        ea_host[u] = hr*esat; // Definition of vapor pressure (see Campbell and Norman pp. 42 Eq. 3.11)

        //Air pressure
        if( context->doesPrimitiveDataExist(p,"air_pressure") && context->getPrimitiveDataType(p,"air_pressure")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"air_pressure",pressure_host[u]);
            if( pressure_host[u]<10000.f ){
                if( message_flag ) {
                    std::cout << "WARNING (EnergyBalanceModel::run): Value of " << pressure_host[u] << " given in 'air_pressure' primitive data is very small. Values should be given in units of Pascals. Assuming default value of " << pressure_default << std::endl;
                }
              pressure_host[u] = pressure_default;
            }
        }else{
            pressure_host[u] = pressure_default;
        }

        //Number of sides emitting radiation
        Nsides_host[u] = 2; //default is 2
        if( context->doesPrimitiveDataExist(p,"twosided_flag") && context->getPrimitiveDataType(p,"twosided_flag")==HELIOS_TYPE_UINT ){
          uint flag;
          context->getPrimitiveData(p,"twosided_flag",flag);
          if( flag==0 ){
            Nsides_host[u]=1;
          }
        }

        //Number of evaporating/transpiring faces
        stomatal_sidedness_host[u] = 0.f; //if Nsides=1, force this to be 0 (all stomata on upper surface)
        if( Nsides_host[u]==2 && context->doesPrimitiveDataExist(p,"stomatal_sidedness") && context->getPrimitiveDataType(p,"stomatal_sidedness")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"stomatal_sidedness",stomatal_sidedness_host[u]);
        //this is for backward compatability prior to v1.3.17
        }else if( Nsides_host[u]==2 && context->doesPrimitiveDataExist(p,"evaporating_faces") && context->getPrimitiveDataType(p,"evaporating_faces")==HELIOS_TYPE_UINT ){
          uint flag;
          context->getPrimitiveData(p,"evaporating_faces",flag);
          if( flag==1 ) { //stomata on one side
            stomatal_sidedness_host[u] = 0.f;
          }else if( flag==2 ){
            stomatal_sidedness_host[u] = 0.5f;
          }
        }

        //Boundary-layer conductance to heat
        if( context->doesPrimitiveDataExist(p,"boundarylayer_conductance") && context->getPrimitiveDataType(p,"boundarylayer_conductance")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"boundarylayer_conductance",gH_host[u]);
        }else{

            //Wind speed
            float U;
            if( context->doesPrimitiveDataExist(p,"wind_speed") && context->getPrimitiveDataType(p,"wind_speed")==HELIOS_TYPE_FLOAT ){
                context->getPrimitiveData(p,"wind_speed",U);
            }else{
                U = wind_speed_default;
            }

            //Characteristic size of primitive
            float L;
            if( context->doesPrimitiveDataExist(p,"object_length") && context->getPrimitiveDataType(p,"object_length")==HELIOS_TYPE_FLOAT ){
                context->getPrimitiveData(p,"object_length",L);
                if( L==0 ){
                    L = sqrt(context->getPrimitiveArea(p));
                    primitive_length_used = true;
                }
            }else if( context->getPrimitiveParentObjectID(p)>0 ){
              uint objID = context->getPrimitiveParentObjectID(p);
              L = sqrt(context->getObjectArea(objID));
            }else{
                L = sqrt(context->getPrimitiveArea(p));
                primitive_length_used = true;
            }

            gH_host[u]=0.135f*sqrt(U/L)*float(Nsides_host[u]);

            calculated_blconductance_used = true;
        }

        //Moisture conductance
        if( context->doesPrimitiveDataExist(p,"moisture_conductance") && context->getPrimitiveDataType(p,"moisture_conductance")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"moisture_conductance",gS_host[u]);
        }else{
            gS_host[u] = gS_default;
        }

        //Other fluxes
        if( context->doesPrimitiveDataExist(p,"other_surface_flux") && context->getPrimitiveDataType(p,"other_surface_flux")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"other_surface_flux",Qother_host[u]);
        }else{
            Qother_host[u] = Qother_default;
        }

        //Object heat capacity
        if( context->doesPrimitiveDataExist(p,"heat_capacity") && context->getPrimitiveDataType(p,"heat_capacity")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"heat_capacity",heatcapacity_host[u]);
        }else{
            heatcapacity_host[u] = heatcapacity_default;
        }

        //Surface humidity
        if( context->doesPrimitiveDataExist(p,"surface_humidity") && context->getPrimitiveDataType(p,"surface_humidity")==HELIOS_TYPE_FLOAT ){
            context->getPrimitiveData(p,"surface_humidity",surfacehumidity_host[u]);
        }else{
            surfacehumidity_host[u] = surface_humidity_default;
        }

        //Emissivity
        eps_host[u] = emissivity.at(u);

        //Net absorbed radiation
        R_host[u] = Rn.at(u);

    }

    //if we used the calculated boundary-layer conductance, enable output primitive data "boundarylayer_conductance_out" so that it can be used by other plug-ins
    if( calculated_blconductance_used ){
        auto it = find( output_prim_data.begin(), output_prim_data.end(), "boundarylayer_conductance_out" );
        if( it == output_prim_data.end() ){
            output_prim_data.emplace_back( "boundarylayer_conductance_out" );
        }
    }

    //if the length of a primitive that is not a member of an object was used, issue a warning
    if( message_flag && primitive_length_used ){
        std::cout << "WARNING (EnergyBalanceModel::run): The length of a primitive that is not a member of a compound object was used to calculate the boundary-layer conductance. This often results in incorrect values because the length should be that of the object (e.g., leaf, stem) not the primitive. Make sure this is what you intended." << std::endl;
    }

    // Create Kokkos Views and copy data to device
    Kokkos::View<float*> d_To("To", Nprimitives);
    Kokkos::View<float*> d_R("R", Nprimitives);
    Kokkos::View<float*> d_Qother("Qother", Nprimitives);
    Kokkos::View<float*> d_eps("eps", Nprimitives);
    Kokkos::View<float*> d_Ta("Ta", Nprimitives);
    Kokkos::View<float*> d_ea("ea", Nprimitives);
    Kokkos::View<float*> d_pressure("pressure", Nprimitives);
    Kokkos::View<float*> d_gH("gH", Nprimitives);
    Kokkos::View<float*> d_gS("gS", Nprimitives);
    Kokkos::View<uint*> d_Nsides("Nsides", Nprimitives);
    Kokkos::View<float*> d_stomatal_sidedness("stomatal_sidedness", Nprimitives);
    Kokkos::View<float*> d_heatcapacity("heatcapacity", Nprimitives);
    Kokkos::View<float*> d_surfacehumidity("surfacehumidity", Nprimitives);
    Kokkos::View<float*> d_T("T", Nprimitives);

    // Copy data from host to device
    auto h_To = Kokkos::create_mirror_view(d_To);
    auto h_R = Kokkos::create_mirror_view(d_R);
    auto h_Qother = Kokkos::create_mirror_view(d_Qother);
    auto h_eps = Kokkos::create_mirror_view(d_eps);
    auto h_Ta = Kokkos::create_mirror_view(d_Ta);
    auto h_ea = Kokkos::create_mirror_view(d_ea);
    auto h_pressure = Kokkos::create_mirror_view(d_pressure);
    auto h_gH = Kokkos::create_mirror_view(d_gH);
    auto h_gS = Kokkos::create_mirror_view(d_gS);
    auto h_Nsides = Kokkos::create_mirror_view(d_Nsides);
    auto h_stomatal_sidedness = Kokkos::create_mirror_view(d_stomatal_sidedness);
    auto h_heatcapacity = Kokkos::create_mirror_view(d_heatcapacity);
    auto h_surfacehumidity = Kokkos::create_mirror_view(d_surfacehumidity);

    for(uint i = 0; i < Nprimitives; i++) {
        h_To(i) = To_host[i];
        h_R(i) = R_host[i];
        h_Qother(i) = Qother_host[i];
        h_eps(i) = eps_host[i];
        h_Ta(i) = Ta_host[i];
        h_ea(i) = ea_host[i];
        h_pressure(i) = pressure_host[i];
        h_gH(i) = gH_host[i];
        h_gS(i) = gS_host[i];
        h_Nsides(i) = Nsides_host[i];
        h_stomatal_sidedness(i) = stomatal_sidedness_host[i];
        h_heatcapacity(i) = heatcapacity_host[i];
        h_surfacehumidity(i) = surfacehumidity_host[i];
    }

    Kokkos::deep_copy(d_To, h_To);
    Kokkos::deep_copy(d_R, h_R);
    Kokkos::deep_copy(d_Qother, h_Qother);
    Kokkos::deep_copy(d_eps, h_eps);
    Kokkos::deep_copy(d_Ta, h_Ta);
    Kokkos::deep_copy(d_ea, h_ea);
    Kokkos::deep_copy(d_pressure, h_pressure);
    Kokkos::deep_copy(d_gH, h_gH);
    Kokkos::deep_copy(d_gS, h_gS);
    Kokkos::deep_copy(d_Nsides, h_Nsides);
    Kokkos::deep_copy(d_stomatal_sidedness, h_stomatal_sidedness);
    Kokkos::deep_copy(d_heatcapacity, h_heatcapacity);
    Kokkos::deep_copy(d_surfacehumidity, h_surfacehumidity);

    // Define functor for energy balance solve
    struct EnergyBalanceSolver {
        Kokkos::View<float*> To, R, Qother, eps, Ta, ea, pressure, gH, gS, stomatal_sidedness, heatcapacity, surfacehumidity, T;
        Kokkos::View<uint*> Nsides;
        float dt;

        EnergyBalanceSolver(Kokkos::View<float*> To_, Kokkos::View<float*> R_, Kokkos::View<float*> Qother_,
                           Kokkos::View<float*> eps_, Kokkos::View<float*> Ta_, Kokkos::View<float*> ea_,
                           Kokkos::View<float*> pressure_, Kokkos::View<float*> gH_, Kokkos::View<float*> gS_,
                           Kokkos::View<uint*> Nsides_, Kokkos::View<float*> stomatal_sidedness_,
                           Kokkos::View<float*> heatcapacity_, Kokkos::View<float*> surfacehumidity_,
                           Kokkos::View<float*> T_, float dt_)
            : To(To_), R(R_), Qother(Qother_), eps(eps_), Ta(Ta_), ea(ea_), pressure(pressure_),
              gH(gH_), gS(gS_), Nsides(Nsides_), stomatal_sidedness(stomatal_sidedness_),
              heatcapacity(heatcapacity_), surfacehumidity(surfacehumidity_), T(T_), dt(dt_) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const uint p) const {
            float Tval;

            float err_max = 0.0001;
            uint max_iter = 100;

            float T_old_old = To(p);

            float T_old = T_old_old;
            T_old_old = 400.f;

            float resid_old = evaluateEnergyBalance(T_old,R(p),Qother(p),eps(p),Ta(p),ea(p),pressure(p),gH(p),gS(p),Nsides(p),stomatal_sidedness(p),heatcapacity(p),surfacehumidity(p),dt,To(p));
            float resid_old_old = evaluateEnergyBalance(T_old_old,R(p),Qother(p),eps(p),Ta(p),ea(p),pressure(p),gH(p),gS(p),Nsides(p),stomatal_sidedness(p),heatcapacity(p),surfacehumidity(p),dt,To(p));

            float resid = 100;
            float err = resid;
            uint iter = 0;
            while( err>err_max && iter<max_iter ){

                if( resid_old==resid_old_old ){//this condition will cause NaN
                    err=0;
                    break;
                }

                Tval = Kokkos::fabs((T_old_old*resid_old-T_old*resid_old_old)/(resid_old-resid_old_old));

                resid = evaluateEnergyBalance(Tval,R(p),Qother(p),eps(p),Ta(p),ea(p),pressure(p),gH(p),gS(p),Nsides(p),stomatal_sidedness(p),heatcapacity(p),surfacehumidity(p),dt,To(p));

                resid_old_old = resid_old;
                resid_old = resid;

                err = Kokkos::fabs(T_old-T_old_old)/Kokkos::fabs(T_old_old);

                T_old_old = T_old;
                T_old = Tval;

                iter++;

            }

            if( err>err_max ){
                Kokkos::printf("WARNING (EnergyBalanceModel::solveEnergyBalance): Energy balance did not converge.\n");
            }

            T(p) = Tval;
        }
    };

    // Launch Kokkos kernel
    EnergyBalanceSolver solver(d_To, d_R, d_Qother, d_eps, d_Ta, d_ea, d_pressure, d_gH, d_gS, d_Nsides, d_stomatal_sidedness, d_heatcapacity, d_surfacehumidity, d_T, dt);
    Kokkos::parallel_for("solveEnergyBalance", Kokkos::RangePolicy<>(0, Nprimitives), solver);

    // Copy results back to host
    auto h_T = Kokkos::create_mirror_view(d_T);
    Kokkos::deep_copy(h_T, d_T);

    // Copy results back to context
    for( uint u=0; u<Nprimitives; u++ ){
        size_t p = UUIDs.at(u);

        float T = h_T(u);

        if( T!=T ){
            T = temperature_default;
        }

        context->setPrimitiveData(p,"temperature",T);

        float QH = 29.25*gH_host[u]*(T-Ta_host[u]);
        context->setPrimitiveData(p,"sensible_flux",QH);

        float es = 611.f*exp(17.502f*(T-273.f)/((T-273.f)+240.97f));
        float gM = 1.08f*gH_host[u]*gS_host[u]*(stomatal_sidedness_host[u]/(1.08f*gH_host[u]+gS_host[u]*stomatal_sidedness_host[u]) + (1.f-stomatal_sidedness_host[u])/(1.08f*gH_host[u]+gS_host[u]*(1.f-stomatal_sidedness_host[u])));
        if( gH_host[u]==0 && gS_host[u]==0 ){//if somehow both go to zero, can get NaN
            gM = 0;
        }
        float QL = 44000*gM*(es-ea_host[u])/pressure_host[u];
        context->setPrimitiveData(p,"latent_flux",QL);

        float storage=0.f;
        if ( dt>0){
            storage=heatcapacity_host[u]*(T-To_host[u])/dt;
        }
        context->setPrimitiveData(p,"storage_flux", storage);

        for( int i=0; i<output_prim_data.size(); i++ ){
            if( output_prim_data.at(i) == "boundarylayer_conductance_out" ){
                context->setPrimitiveData(p,"boundarylayer_conductance_out",gH_host[u]);
            }else if( output_prim_data.at(i) == "vapor_pressure_deficit" ){
                float vpd = (es-ea_host[u])/pressure_host[u];
                context->setPrimitiveData(p,"vapor_pressure_deficit",vpd);
            }
        }

    }

    if( message_flag ){
        std::cout << "done." << std::endl;
    }

}
