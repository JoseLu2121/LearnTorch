#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/tensor.h"
#include "../include/ops.h"
#include "../include/optimizer.h"
#include "../include/layers.h"
#include "../include/structure.h"
#include "../include/loss.h"
#include "../include/trainer.h"

namespace py = pybind11;

PYBIND11_MODULE(learntorch, m) {
    m.doc() = "Binding de Learntorch C++";

    // --- TENSOR ---
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<const std::vector<int>&, const std::vector<float>&>(), 
             py::arg("shape"), py::arg("data") = std::vector<float>{})
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def_readwrite("grad", &Tensor::grad)
        .def("backward", &Tensor::backward)
        .def("item", [](std::shared_ptr<Tensor> t) { return t->getData()[0]; }) 
        .def("data", [](std::shared_ptr<Tensor> t) { 
             return std::vector<float>(t->getData(), t->getData() + t->getSize());
        })
        
        // Operadores Mágicos
        .def("__add__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return a + b; })
        .def("__add__", [](std::shared_ptr<Tensor> a, float val) { return a + val; }) 
        .def("__sub__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return a - b; })
        .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return a * b; })
        .def("__mul__", [](std::shared_ptr<Tensor> a, float val) { return a * val; })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return a / b; })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, float val) { return a / val; })
        .def("__matmul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return matmul(a, b); })
        
        .def("__repr__", [](const Tensor &t) {
            std::string s = "Tensor(shape=[";
            for(size_t i=0; i<t.shape.size(); i++) s += (i>0?",":"") + std::to_string(t.shape[i]);
            s += "])"; 
            return s;
        });

    // --- BLOCKS & LAYERS ---
    py::class_<Block, std::shared_ptr<Block>>(m, "Block")
        .def("parameters", &Block::parameters)
        .def("forward", &Block::forward);

    py::class_<Linear, Block, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int>(), py::arg("in_features"), py::arg("out_features"))
        .def_readwrite("W", &Linear::W)
        .def_readwrite("B", &Linear::B);
    
    py::class_<ReLU, Block, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>());

    // Serial (Sequential)
    py::class_<Serial, Block, std::shared_ptr<Serial>>(m, "Serial")
        .def(py::init<const std::vector<std::shared_ptr<Block>>&>());

    py::class_<Softmax, Block, std::shared_ptr<Softmax>>(m,"Softmax")
        .def(py::init<>());

    // --- OPTIMIZADORES ---
    py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer");
    
    // Convertimos automáticamente la lista de Python a vector<shared_ptr<Tensor>>
    py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float>(), 
             py::arg("params"), py::arg("lr"))
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);

    // --- LOSSES ---
    py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss")
        .def("forward", &Loss::forward);
    
    py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init<>());
    
    py::class_<CrossEntropy, Loss, std::shared_ptr<CrossEntropy>>(m, "CrossEntropy")
        .def(py::init<>());

    // --- TRAINER ---
    py::class_<Trainer>(m, "Trainer")
        .def(py::init<std::shared_ptr<Block>, std::shared_ptr<Optimizer>, std::shared_ptr<Loss>>())
        .def("fit", &Trainer::fit, py::arg("x_train"), py::arg("y_train"), py::arg("epochs"), py::arg("print_every")=1);


    // --- FUNCIONES GLOBALES ---
    m.def("matmul", &matmul);
    m.def("relu", &relu);
    m.def("log", [](std::shared_ptr<Tensor> t) { return log(t); });
    m.def("exp", [](std::shared_ptr<Tensor> t) { return exp(t); });
    m.def("sum", &sum, py::arg("tensor"), py::arg("dim")=0);
    m.def("transpose", &transpose_view);
}