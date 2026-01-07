/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

 * SYCL Quick Reference
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * // Default construct a queue
 * auto q = sycl::queue{};
 *
 * // Declare a buffer pointing to ptr
 * auto buf = sycl::buffer{ptr, sycl::range{n}};
 *
 * // Submit work to the queue
 * q.submit([&](sycl::handler &cgh) {
 *   // COMMAND GROUP
 * });
 *
 * // Within the command group you can
 * //    1. Declare an accessor to a buffer
 *          auto read_write_acc = sycl::accessor{buf, cgh};
 *          auto read_acc = sycl::accessor{buf, cgh, sycl::read_only};
 *          auto write_acc = sycl::accessor{buf, cgh, sycl::write_only};
 *          auto no_init_acc = sycl::accessor{buf, cgh, sycl::no_init};
 * //    2. Enqueue a single task:
 *          cgh.single_task<class mykernel>([=]() {
 *              // Do something
 *          });
 * //    3. Enqueue a parallel for:
 *          cgh.parallel_for<class mykernel>(sycl::range{n}, [=](sycl::id<1> i)
 {
 *              // Do something
 *          });
 *
*/

#include <exception>
#include "../helpers.hpp"

#include <sycl/nd_item.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/sycl.hpp>

int main() {
  constexpr size_t dataSize = 1024;

  int a[dataSize], b[dataSize], r[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = i;
    b[i] = i;
    r[i] = 0;
  }

  // Task: Compute r[i] = a[i] + b[i] in parallel on the SYCL device
  // for (int i = 0; i < dataSize; ++i) {
  //   r[i] = a[i] + b[i];
  // }
  try{
    auto q = sycl::queue([=](sycl::exception_list el){
      for (auto e: el){
        std::rethrow_exception(e);
      }
    });

    auto buf_a = sycl::buffer(a, sycl::range<1>(dataSize));
    auto buf_b = sycl::buffer(b, sycl::range<1>(dataSize));
    auto buf_r = sycl::buffer(r, sycl::range<1>(dataSize));

    q.submit([&](sycl::handler &cgh){
      auto acc_a = sycl::accessor(buf_a, cgh);
      auto acc_b = sycl::accessor(buf_b, cgh);
      auto acc_r = sycl::accessor(buf_r, cgh);

      // simple sycl::range based kernel
      cgh.parallel_for(sycl::range<1>(dataSize), [=](sycl::id<1> i){
        acc_r[i] = acc_a[i] + acc_b[i];
      });

      // simple size_t based kernel
      // cgh.parallel_for(dataSize, [=](sycl::id<1> i){
      //   acc_r[i] = acc_a[i] + acc_b[i];
      // });

      // sycl::nd_range based kernel (potentially deprecated)
      // cgh.parallel_for(sycl::nd_range<1>(dataSize, 1), [=](sycl::nd_item<1> item){
      //   auto i = item.get_global_linear_id();
      //   acc_r[i]  = acc_a[i] + acc_b[i];
      // }); 
    }).wait();
  } catch (std::exception &e){
    std::cout << "Error: " << e.what() << std::endl;
  }

  // no need to copy back the data since buffer automatically does so to host
  // when destroyed (provided it was initialized using host memory)

  SYCLACADEMY_ASSERT_EQUAL(r, [](size_t i) { return i * 2; });
}
