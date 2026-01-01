/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.

 * SYCL Quick Reference
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * // Include SYCL header
 * #include <sycl/sycl.hpp>
 *
 * // Default construct a queue
 * auto q = sycl::queue{};
 *
 * // Allocate device memory
 * auto * devPtr = sycl::malloc_device<int>(mycount, q);
 *
 * // Memcpy
 * q.memcpy(dst, src, sizeof(T)*n).wait();
 * // (dst and src are pointers)
 *
 * // Free memory
 * sycl::free(ptr, q);
 *
 * // Construct a buffer of size n associated with ptr
 * auto buf = sycl::buffer{ptr, sycl::range{n}};
 *
 * // Submit a kernel
 * q.submit([&](sycl::handler &cgh) {
 *    cgh.single_task([=](){
 *      // Some kernel code
 *      });
 * }).wait();
 *
 * // Construct an accessor for buf
 * // (must be done within command group)
 *     auto acc = sycl::accessor{buf, cgh};
 *     auto acc = sycl::accessor{buf, cgh, sycl::read_only};
 *     auto acc = sycl::accessor{buf, cgh, sycl::write_only};
 *     auto acc = sycl::accessor{buf, cgh, sycl::no_init};
 *
*/

#include "../helpers.hpp"

#include <sycl/sycl.hpp>

void test_usm() {
  int a = 18, b = 24, r = 0;

  // Task: Compute a+b on the SYCL device using USM
  // r = a + b;
  auto q = sycl::queue();

  auto a_dptr = sycl::malloc_device<int>(sizeof(a), q);
  auto b_dptr = sycl::malloc_device<int>(sizeof(b), q);
  auto r_dptr = sycl::malloc_device<int>(sizeof(r), q);

  q.memcpy(a_dptr, &a, sizeof(a));
  q.memcpy(b_dptr, &b, sizeof(b));

  q.single_task([=]() {
    r_dptr[0] = a_dptr[0] + b_dptr[0];
  }).wait();  // wait is required since unlike streams in CUDA, queues in SYCL are not in-order.  
  
  q.memcpy(&r, r_dptr, sizeof(r)).wait();

  SYCLACADEMY_ASSERT_EQUAL(r, 42);

  sycl::free(a_dptr, q);
  sycl::free(b_dptr, q);
  sycl::free(r_dptr, q);
}

void test_buffer() {
  int a = 18, b = 24, r = 0;

  // Task: Compute a+b on the SYCL device using the buffer
  // accessor memory model
  // r = a + b;

  auto a_buf = sycl::buffer(&a, sycl::range<1>(1));
  auto b_buf = sycl::buffer(&b, sycl::range<1>(1));
  auto r_buf = sycl::buffer(&r, sycl::range<1>(1));

  auto q = sycl::queue();
  q.submit([&](sycl::handler &cgh) {
    auto a_acc = sycl::accessor(a_buf, cgh, sycl::read_only);
    auto b_acc = sycl::accessor(b_buf, cgh, sycl::read_only);
    auto r_acc = sycl::accessor(r_buf, cgh, sycl::write_only);

    cgh.single_task([=](){
      r_acc[0] = a_acc[0] + b_acc[0];
    });
  }).wait();

  SYCLACADEMY_ASSERT_EQUAL(r, 42);
}

int main() {
  test_usm();
  test_buffer();
}
