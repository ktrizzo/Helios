#include "EnergyBalanceModel.h"
#include <Kokkos_Core.hpp>

using namespace helios;

int main(){

    Kokkos::initialize();

    Context context;

    EnergyBalanceModel energybalance( &context );

    int result = energybalance.selfTest();

    Kokkos::finalize();

    return result;

}
