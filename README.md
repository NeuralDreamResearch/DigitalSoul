# Digital Soul
### Unified Compute Platform - CPU, GPU, FPGA, Quantum Computing

<img src="https://github.com/NeuralDreamResearch/DigitalSoulPy/blob/main/Logo.png?raw=true" height=300>

DigitalSoul is a Python module designed to bridge the gap between classical, quantum, and potentially hardware-accelerated computation. It provides flexible data structures and a node-based execution model, allowing you to express computations that can be seamlessly executed across CPU, GPU, quantum simulators, and potentially FPGAs.

## Key Features

*   **Customizable Data Types:** Define Boolean (Bool), integer (Int, UInt), floating-point (Float), quantum states (Qudit), quantum gates (QuantumGate), and multidimensional tensors (Tensor) to suit your computational needs.
*   **Node-Based Computation:** Build computational graphs using nodes that represent operations (e.g., LogicalAnd, LogicalOr).  Nodes manage input/output data through "Edges".
*   **Multi-Backend Execution:** Execute computations using NumPy, Cupy (for GPU), TensorFlow, and potentially quantum simulators like Qiskit.
*   **VHDL Transpilation:** Translate computational graphs into VHDL code, opening the door for hardware synthesis on FPGAs.

## Quick Example

```python
import DigitalSoul as ds

e1=ds.Edge(ds.Bool(False))
e2=ds.Edge(ds.Bool(True))
e3=ds.Edge(ds.Bool(None))
e4=ds.Edge(ds.Bool(True))
e5=ds.Edge(ds.Bool(None))
e6=ds.Edge(ds.Bool(False))
e7=ds.Edge(ds.Bool(None))

print("\n"*4)
or_gate=ds.LogicalOr((e1,e2), e3)
or_gate=ds.LogicalOr((e3,e4), e5)
and_gate1=ds.LogicalAnd((e5,e6), e7)
print(e7)
print("Executing function")
and_gate1.execute("cp")
print(e7)
print("\n"*3)
print(and_gate1.transpile("vhdl"))
```

## Roadmap

*   Full integration with Qiskit and/or QALU for quantum computation.
*   Improved hardware synthesis flow with VHDL transpilation.
*   Custom node creation guide.

## Contributing

We welcome contributions to DigitalSoul!

## License

DigitalSoul is distributed under the MIT License (see LICENSE.md).
