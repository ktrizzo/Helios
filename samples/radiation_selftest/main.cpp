#include "RadiationModel.h"

#ifdef HELIOS_USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

using namespace helios;

int main()
{

#ifdef HELIOS_USE_KOKKOS
  Kokkos::initialize();
#endif

  Context context;

  RadiationModel radiationmodel(&context);

  int result = radiationmodel.selfTest();

#ifdef HELIOS_USE_KOKKOS
  Kokkos::finalize();
#endif

  return result;

}
