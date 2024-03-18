#include <pybind11/pybind11.h> 
#include "DigitalSoulCpp.hpp" // Your main header file
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper function to avoid code repetition in bindings
template <typename T>
void expose_QN_class(py::module& m, const std::string& name) {
    py::class_<QN::template Complex<T>>(m, name.c_str())
        .def(py::init<T, T>())
        .def("__call__", &QN::template Complex<T>::operator()) 
        .def("__add__", &QN::template Complex<T>::operator+)
        .def("__sub__", &QN::template Complex<T>::operator-)
        .def("__mul__", &QN::template Complex<T>::operator*)
        .def("__truediv__", &QN::template Complex<T>::operator/) // Use __truediv__ for Python 3+
        .def("magnitude", &QN::template Complex<T>::magnitude)
        .def("arg", &QN::template Complex<T>::arg)
        .def("conj", &QN::template Complex<T>::conj)
        .def("__str__", &QN::template Complex<T>::ss); // Note: ss should return a std::string
}

template <typename T>
void expose_QN_class_qudit(py::module& m, const std::string& name) {
    py::class_<QN::template Qudit<T>>(m, name.c_str())
        .def(py::init<size_t, bool>())
        .def(py::init<size_t, QN::template Complex<T>*, T, bool>()) // Trust constructor option
        .def("oneHot", &QN::template Qudit<T>::oneHot)
        .def("loadStatevector", &QN::template Qudit<T>::loadStatevector)
        .def("freeStatevector", &QN::template Qudit<T>::freeStatevector)
        .def("Psi", &QN::template Qudit<T>::Psi)
        .def("numStates", &QN::template Qudit<T>::numStates)
        .def("__str__", &QN::template Qudit<T>::ss); // Note: ss should return a std::string
}

template <typename T>
void expose_QN_class_gate(py::module& m, const std::string& name) {
    py::class_<QN::template Gate<T>>(m, name.c_str())
        .def(py::init<size_t>())
        .def(py::init<size_t, QN::template Complex<T>*, T, bool>()) // Trust constructor option
        .def("loadOperator", &QN::template Gate<T>::loadOperator)
        .def("transform", &QN::template Gate<T>::transform)
        .def("__str__", &QN::template Gate<T>::ss); // Note: ss should return a std::string
}

PYBIND11_MODULE(dscpp, m) {
    m.doc() = "DigitalSoul Python Bindings"; // Optional module docstring

    // QN::Complex
    expose_QN_class<float>(m, "Complex");

    // QN::Qudit
    expose_QN_class_qudit<float>(m, "Qudit");

    // QN::Gate
    expose_QN_class_gate<float>(m, "Gate");


    // Expose LUTx_1 class
    py::class_<LUTx_1>(m, "LUTx_1")
        .def(py::init<size_t, size_t>())
        .def("getNumInputs", &LUTx_1::getNumInputs)
        .def("getLogicID", &LUTx_1::getLogicID)
        .def("setNumInputs", &LUTx_1::setNumInputs)
        .def("setLogicID", &LUTx_1::setLogicID)
        .def("computeCPU", &LUTx_1::computeCPU)
        .def("UnitaryGen", &LUTx_1::UnitaryGen<float>) // Use the templated version
        .def("entityGen", &LUTx_1::entityGen)
        .def("ss", &LUTx_1::ss) // Added the ss method
        .def("LookUpTable", &LUTx_1::LookUpTable)  // Added the LookUpTable method
        .def("ThermoTable", &LUTx_1::ThermoTable);
}

