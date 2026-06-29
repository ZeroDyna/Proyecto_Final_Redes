#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <list>
#include "ucsp.hpp"
#include "nn_config.hpp"

using namespace std;

// Alias
namespace py = pybind11;

py::array_t<float> distribute_wrapper(py::list buffers_per_worker, py::list workers_list, size_t model_params) {
  // n: # de workers
  int n = py::len(workers_list);
  //cout << n << endl;
  vector<py::array_t<float>> kept;

  // Esto ya esta asegurado, si quieren lo quitan nomas
  if (n <= 0) {
    throw runtime_error("We need at least one worker");
  }

  if (py::len(buffers_per_worker) != n) {
    throw runtime_error("buffers_per_worker and workers must have the same size");
  }

  
  kept.reserve(n);
  vector<float *> data_in(n);
  vector<size_t> elements(n);

  for (int i = 0; i < n; i++) {
    py::array_t<float, py::array::c_style | py::array::forcecast> arr = buffers_per_worker[i].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    kept.push_back(arr);
    py::buffer_info info = arr.request();

    // NO LE QUITEN LOS CASTS O EXPLOTARA
    data_in[i] = (float *)info.ptr;
    elements[i] = info.size;
  }

  vector<string> ip_storage(n);
  vector<WorkerInfo> workers(n);

  for (int i = 0; i < n; i++) {
    py::tuple w = workers_list[i].cast<py::tuple>();
    ip_storage[i] = w[0].cast<string>();
    workers[i].ip = ip_storage[i].c_str();
    workers[i].port = w[1].cast<int>();
  }

  py::array_t<float> gradient_avg(model_params);
  py::buffer_info out_info = gradient_avg.request();
  float *out_ptr = (float *)out_info.ptr;

  // Bug catching
  int rc = ucsp_distribute(data_in.data(), elements.data(), workers.data(),n, out_ptr, model_params);
  //cout << rc << endl;
  if (rc != 0) {
    throw runtime_error("BUG DETECTED!: " + to_string(rc));
  }
  return gradient_avg;
}

PYBIND11_MODULE(ucsp, m) {
  m.doc() = "Binding UCSP (UDP + RDT)";
  m.def("distribute", &distribute_wrapper, py::arg("buffers_per_worker"), py::arg("workers"), py::arg("model_params") = (size_t)TOTAL_MODEL_PARAMS,"");
  m.attr("TOTAL_MODEL_PARAMS") = TOTAL_MODEL_PARAMS;
  m.attr("INPUT_DIM") = INPUT_DIM;
  m.attr("NUM_CLASSES") = NUM_CLASSES;
}
