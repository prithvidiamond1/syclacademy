/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include <exception>
#include <sycl/exception_list.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>
#include "../helpers.hpp"

int main() {
  constexpr size_t dataSize = 1024;

  float a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
    r[i] = 0.0f;
  }

  // Task: Allocate the arrays in USM, and compute r[i] = a[i] + b[i] on the
  // SYCL device
  // for (int i = 0; i < dataSize; ++i) {
  //   r[i] = a[i] + b[i];
  // }

  try{
    // custom device selector to get USM device using usm_device_allocations aspect
    auto q = sycl::queue([](const sycl::device &dev) -> int { return (dev.has(sycl::aspect::usm_device_allocations) ? 1 : 0); },
    [=](sycl::exception_list el){
      for(auto e: el){
        std::rethrow_exception(e);
      }
    });
    auto dev_a = sycl::malloc_device<float>(sizeof(float)*dataSize, q);
    auto dev_b = sycl::malloc_device<float>(sizeof(float)*dataSize, q);
    auto dev_r = sycl::malloc_device<float>(sizeof(float)*dataSize, q);

    q.memcpy(dev_a, a, sizeof(float)*dataSize);
    q.memcpy(dev_b, b, sizeof(float)*dataSize);

    q.parallel_for(sycl::range<1>(dataSize), [=](sycl::id<1> id){
      dev_r[id] = dev_a[id] + dev_b[id];
    }).wait();

    q.memcpy(r, dev_r, sizeof(float)*dataSize);
    
    sycl::free(dev_a, q);
    sycl::free(dev_b, q);
    sycl::free(dev_r, q);
  } catch (std::exception &e){
    std::cout << "Error: " << e.what() << std::endl;
  }

  SYCLACADEMY_ASSERT_EQUAL(r, [](size_t i) { return i * 2.0f; });
}
