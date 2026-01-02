/*
 SYCL Academy (c)

 SYCL Academy is licensed under a Creative Commons
 Attribution-ShareAlike 4.0 International License.

 You should have received a copy of the license along with this
 work.  If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
*/

#include <exception>
#include <sycl/aspects.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/sycl.hpp>

#include "../helpers.hpp"

int main() {
  // Task: catch synchronous and asynchronous exceptions

  try{
    auto defaultQueue = sycl::queue{
       [=](sycl::exception_list el) {
        for (auto e: el) {
          std::rethrow_exception(e);
        }
    }};

    auto buf = sycl::buffer<int>(sycl::range{1});

    defaultQueue
        .submit([&](sycl::handler& cgh) {
          // This throws an exception: an accessor has a range which is
          // outside the bounds of its buffer.
          auto acc = buf.get_access(cgh, sycl::range{2}, sycl::read_write);
        })
        .wait();

    defaultQueue.throw_asynchronous();
  } catch (const std::exception &e){
    std::cout << "Error: " << e.what() << std::endl;
  }
  
  SYCLACADEMY_ASSERT(true);
}
