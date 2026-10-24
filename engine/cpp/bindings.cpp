// The C++ engine as a Python module, cfr_cpp. engine/native.py builds and imports it.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cfr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cfr_cpp, m) {
    py::class_<cfr::Spot>(m, "Spot")
        .def(py::init([](std::string ranks, std::string suits, std::vector<std::string> board, int blind,
                         int bet, int max_raises, std::optional<int> stack, std::vector<double> pot_sizes) {
                 return cfr::Spot{ranks, suits, board, blind, bet, max_raises, stack, pot_sizes};
             }),
             py::kw_only(), py::arg("ranks"), py::arg("suits"), py::arg("board"), py::arg("blind"),
             py::arg("bet"), py::arg("max_raises"), py::arg("stack"), py::arg("pot_sizes"));

    py::class_<cfr::Child>(m, "Child")
        .def_readonly("node", &cfr::Child::node)
        .def_readonly("folder", &cfr::Child::folder)
        .def_readonly("contrib", &cfr::Child::contrib);

    py::class_<cfr::Node>(m, "Node")
        .def_readonly("seq", &cfr::Node::seq)
        .def_readonly("player", &cfr::Node::player)
        .def_readonly("contrib", &cfr::Node::contrib)
        .def_readonly("actions", &cfr::Node::actions)
        .def_readonly("children", &cfr::Node::children);

    py::class_<cfr::Solver>(m, "Solver")
        .def(py::init<cfr::Spot>(), py::arg("spot"))
        .def_property_readonly("hands", &cfr::Solver::hands)
        .def_property_readonly("nodes", &cfr::Solver::nodes);
}
