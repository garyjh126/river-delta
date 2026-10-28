// The C++ engine as a Python module, cfr_cpp. engine/native.py builds and imports it.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cards.hpp"
#include "cfr.hpp"

namespace py = pybind11;

namespace {

// A copy of row-major data as a numpy array of shape (rows, cols).
py::array_t<double> matrix(const std::vector<double>& data, size_t rows, size_t cols) {
    py::array_t<double> out({rows, cols});
    std::copy(data.begin(), data.end(), out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(cfr_cpp, m) {
    m.def("hand_value", &cards::value, py::arg("cards"));

    py::class_<cfr::Spot>(m, "Spot")
        .def(py::init([](std::string ranks, std::string suits, std::vector<std::string> board, int blind,
                         int bet, int max_raises, std::optional<int> stack, std::vector<double> pot_sizes,
                         std::optional<std::vector<double>> raise_sizes, std::vector<std::vector<double>> ranges,
                         int streets) {
                 return cfr::Spot{ranks, suits, board, blind, bet, max_raises, stack, pot_sizes, raise_sizes,
                                  ranges, streets};
             }),
             py::kw_only(), py::arg("ranks"), py::arg("suits"), py::arg("board"), py::arg("blind"),
             py::arg("bet"), py::arg("max_raises"), py::arg("stack"), py::arg("pot_sizes"), py::arg("raise_sizes"),
             py::arg("ranges"), py::arg("streets"));

    py::class_<cfr::Child>(m, "Child")
        .def_readonly("node", &cfr::Child::node)
        .def_readonly("chance", &cfr::Child::chance)
        .def_readonly("folder", &cfr::Child::folder)
        .def_readonly("contrib", &cfr::Child::contrib)
        .def_readonly("board", &cfr::Child::board);

    py::class_<cfr::Node>(m, "Node")
        .def_readonly("seq", &cfr::Node::seq)
        .def_readonly("player", &cfr::Node::player)
        .def_readonly("contrib", &cfr::Node::contrib)
        .def_readonly("actions", &cfr::Node::actions)
        .def_readonly("children", &cfr::Node::children)
        .def_readonly("dealt", &cfr::Node::dealt);

    py::class_<cfr::Chance>(m, "Chance")
        .def_readonly("seq", &cfr::Chance::seq)
        .def_readonly("contrib", &cfr::Chance::contrib)
        .def_readonly("dealt", &cfr::Chance::dealt)
        .def_readonly("cards", &cfr::Chance::cards)
        .def_readonly("children", &cfr::Chance::children);

    py::class_<cfr::Board>(m, "Board")
        .def_readonly("dealt", &cfr::Board::dealt)
        .def_readonly("order", &cfr::Board::order)
        .def_readonly("group_start", &cfr::Board::group_start)
        .def_readonly("group_end", &cfr::Board::group_end);

    py::class_<cfr::Step> step(m, "Step");
    py::enum_<cfr::Step::Kind>(step, "Kind")
        .value("visit", cfr::Step::Kind::visit)
        .value("end", cfr::Step::Kind::end)
        .value("update", cfr::Step::Kind::update);
    step.def_readonly("kind", &cfr::Step::kind)
        .def_readonly("iteration", &cfr::Step::iteration)
        .def_readonly("player", &cfr::Step::player)
        .def_readonly("node", &cfr::Step::node)
        .def_readonly("action", &cfr::Step::action)
        .def_readonly("own", &cfr::Step::own)
        .def_readonly("opp", &cfr::Step::opp)
        .def_readonly("strategy", &cfr::Step::strategy)
        .def_readonly("values", &cfr::Step::values)
        .def_readonly("value", &cfr::Step::value)
        .def_readonly("regret", &cfr::Step::regret)
        .def_readonly("strategy_sum", &cfr::Step::strategy_sum);

    py::class_<cfr::Solver>(m, "Solver")
        .def(py::init<cfr::Spot, std::string>(), py::arg("spot"), py::arg("method") = "cfr+")
        .def_property_readonly("hands", &cfr::Solver::hands)
        .def_property_readonly("nodes", &cfr::Solver::nodes)
        .def_property_readonly("chances", &cfr::Solver::chances)
        .def_property_readonly("boards", &cfr::Solver::boards)
        .def_property_readonly("compatible", [](const cfr::Solver& s) {
            return matrix(s.compatible(), s.hands().size(), s.hands().size());
        })
        .def_property_readonly("showdown", [](const cfr::Solver& s) -> py::object {
            if (s.showdown().empty()) {
                return py::none();  // more than one street
            }
            return matrix(s.showdown(), s.hands().size(), s.hands().size());
        })
        .def_property_readonly("ranges", [](const cfr::Solver& s) {
            std::vector<double> both = s.ranges()[0];
            both.insert(both.end(), s.ranges()[1].begin(), s.ranges()[1].end());
            return matrix(both, 2, s.hands().size());
        })
        .def_property_readonly("deal_prob", &cfr::Solver::deal_prob)
        .def_property_readonly("iterations", &cfr::Solver::iterations)
        .def("iterate", &cfr::Solver::iterate, py::arg("iterations"))
        .def("trace", &cfr::Solver::trace, py::arg("iterations"))
        .def("average", [](const cfr::Solver& s, int node) {
            return matrix(s.average(node), s.nodes().at(node).actions.size(), s.hands().size());
        }, py::arg("node"))
        .def("value", &cfr::Solver::value, py::arg("player"))
        .def("best_response", &cfr::Solver::best_response, py::arg("player"))
        .def("exploitability", &cfr::Solver::exploitability);
}
