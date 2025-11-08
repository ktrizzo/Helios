#include "RadiationModel.h"

#ifdef HELIOS_USE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

using namespace helios;

int main(int argc, char* argv[])
{

#ifdef HELIOS_USE_KOKKOS
  Kokkos::initialize(argc, argv);
  std::cout << "Kokkos initialized: " << Kokkos::is_initialized() << std::endl;
#endif

  Context context;

  RadiationModel radiationmodel(&context);

  int result = radiationmodel.selfTest();

#ifdef HELIOS_USE_KOKKOS
  Kokkos::finalize();
#endif

  return result;

}
