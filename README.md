# Digital Soul
### Unified Compute Platform - CPU, GPU, FPGA, Quantum Computing

<img src="https://github.com/NeuralDreamResearch/DigitalSoulPy/blob/main/Logo.png?raw=true" height=300>

DigitalSoul is a Python module designed to bridge the gap between classical, quantum, and potentially hardware-accelerated computation. It provides flexible data structures and a node-based execution model, allowing you to express computations that can be seamlessly executed across CPU, GPU, quantum simulators, and potentially FPGAs.

## Key Features

*   **Customizable Data Types:** Define Boolean (Bool), integer (Int, UInt), floating-point (Float), quantum states (Qudit), quantum gates (QuantumGate), and multidimensional tensors (Tensor) to suit your computational needs.
*   **Node-Based Computation:** Build computational graphs using nodes that represent operations (e.g., LogicalAnd, LogicalOr).  Nodes manage input/output data through "Edges".
*   **Multi-Backend Execution:** Execute computations using NumPy, Cupy (for GPU), TensorFlow, and internal quantum simulator
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
and_gate1.q_stream()
print("\n",e7.sv)
print("\n"*3)
print(and_gate1.transpile("vhdl"))
```
output:
<pre>
Edge_6 holding Bool_6 value=None entropy=1
Executing function
Edge_6 holding Bool_6 value=False entropy=0

 2-levelQudit_3 value=[1. 0.] entropy=0


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
entity main is
Port(
	Bool_3:in std_logic;
	Bool_5:in std_logic;
	Bool_1:in std_logic;
	Bool_0:in std_logic;
	Bool_6:out std_logic
);
end main;
architecture Behavioral of main is
	signal Bool_4:std_logic;
	signal Bool_2:std_logic;
begin
	Bool_4<=Bool_2 or Bool_3;
	Bool_2<=Bool_0 or Bool_1;
	Bool_6<=Bool_4 and Bool_5;
end architecture Behavioral;
</pre>
As you can see from output, the value of output edge(e7, shown as Edge_6) is uncertain before computation. Hence, it has maximum entropy. As soon as value is computed and certainly known, the entropy is zero. Then, program is capable of generating VHDL code of corresponding computational graph. Additionally, it simulated quantum equivalent of the computaitonal graph with Non-Hermetian Gates
## Tree
<pre>
|----Bool-------------------|
|                           |---__init__(value=None)
|                           |---entropy()
|                           |---name()
|                           |---__repr__()
|
|
|----Int--------------------|
|                           |---__init__(value=None,depth=32)
|                           |---bounds()
|                           |---entropy()
|                           |---name()
|                           |---__repr__()
|
|
|----UInt-------------------|
|                           |---__init__(value=None,depth=32)
|                           |---entropy()
|                           |---name()
|                           |---__repr__()
|
|
|----Float------------------|
|                           |---__init__(value=None,exponent=8,mantissa=23)
|                           |---float_info()
|                           |---entropy()
|                           |---name()
|                           |---__repr__()
|
|
|----Qudit------------------|
|                           |---__init__(value,num_levels=None,utol=1e-9)
|                           |---num_levels()
|                           |---entropy()
|                           |---name()
|                           |---__repr__()
|                           |---__and__(other)
|
|
|----QuantumGate(object)----|
|                           |---__init__(data,utol=1e-8)
|                           |---__repr__()
|                           |---data()
|                           |---value()
|                           |---set_data(data,utol)
|                           |---num_levels()
|                           |---entropy()
|                           |---name()
|                           |---__and__(other)
|                           |---__call__(sv)
|
|
|----NonHermitianGate-------|
|                           |---__init__(data)
|                           |---value()
|                           |---name()
|                           |---__call__(sv)
|                           |---__repr__()
|
|
|----Tensor(object)---------|
|                           |---__init__(value,dtype=Float(0),shape=(1,))
|                           |---entropy()
|                           |---__repr__()
|                           |---name()
|
|
|----Edge(object)-----------|
|                           |---__init__(sculk)
|                           |---vhdl()
|                           |---twoscpl(x
|                           |---name()
|                           |---__repr__()
|                           |---entropy()
|                           |---set_predecessor(node)
|                           |---unpack(executor="np")
|                           |---q_info()
|
|
|----Node(object)-----------|
|                           |---__init__(in_terminals,out_terminals,ops={"np"
|                           |---execute(executor="np")
|                           |---edge_accumulator()
|                           |---input_accumulator()
|                           |---node_accumulator()
|                           |---transpile(target="vhdl")
|                           |---qv_contemplate()
|                           |---q_stream()
|
|
|----LogicalAnd(Node)-------|
|
|
|----LogicalOr(Node)--------|
|
|
|----QN---------------------|
                            |---i
                            |---x
                            |---y
                            |---z
                            |---h
                            |---cx
                            |---ccx

  
</pre>
## Roadmap

*   Implementing more nodes
*   Improved hardware synthesis flow with VHDL transpilation.
*   Custom node creation guide.

## Contributing

We welcome contributions to DigitalSoul!

## License

DigitalSoul is distributed under the MIT License (see LICENSE.md).
