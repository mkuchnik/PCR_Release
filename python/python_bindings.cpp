#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pcr.h"

namespace py = pybind11;

std::pair< std::vector< py::bytes >, std::vector<int> > load_PCR_py(
    std::string filename,
    std::vector<int> record_offsets,
    int n_scans
) {
  auto pair = load_PCR(filename, record_offsets, n_scans);
  std::vector<py::bytes> converted_X;
  converted_X.reserve(pair.first.size());
  for (auto& x: pair.first) {
    auto x_bytes = py::bytes(x);
    converted_X.push_back(x_bytes);
  }
  auto pair2 = std::make_pair(converted_X, pair.second);
  return pair2;
}

PYBIND11_MODULE(python_bindings, m) {
  m.doc() = "PCR Bindings for Python3";
  m.def("load_PCR",
        &load_PCR_py,
        "Load Optimized PCR");
}