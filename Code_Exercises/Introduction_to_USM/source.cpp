/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include <exception>
#include <ostream>
#include <sycl/aspects.hpp>
#include <sycl/exception.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/sycl.hpp>

#include "../helpers.hpp"

int main() {
  // Task: create a queue to a device which supports USM allocations
  // Remember to check for exceptions
  try{
    auto usmQueue = sycl::queue{[=](sycl::exception_list el){
      for (auto e: el){
        std::rethrow_exception(e);
      }
    }};

    auto dev = usmQueue.get_device();
    std::cout << "Device Name: " << (dev.get_info<sycl::info::device::name>()) << std::endl;
    std::cout << "USM Device Allocations: " << (dev.has(sycl::aspect::usm_device_allocations) ? "True" : "False") << std::endl;
    std::cout << "USM Host Allocations: " << (dev.has(sycl::aspect::usm_host_allocations) ? "True" : "False") << std::endl;
    std::cout << "USM Shared Allocations: " << (dev.has(sycl::aspect::usm_shared_allocations) ? "True" : "False") << std::endl;
    std::cout << "USM System Allocations: " << (dev.has(sycl::aspect::usm_system_allocations) ? "True" : "False") << std::endl;
  } catch (std::exception &e){
    std::cout << "Error: " << e.what() << std::endl;
  }

  SYCLACADEMY_ASSERT(true);
}
