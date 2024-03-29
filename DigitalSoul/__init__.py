# DigitalSoul: Unified platform for CPU, GPU, FPGA, Quantum Computing
print("Interacting with reality...")
import numpy as np
import DigitalSoul.erg
import DigitalSoul.dscpp


try:
    import tensorflow as tf
    print("Tensorflow is available")
except:
    print("Tensorflow is skipped")
    tf=np    
try:
    import cupy as cp
    print("GPU resources are available. Eager executor can access it.")

except:
    print("GPU resources are not available")
    cp=np

class Fault:
    def __init__(self, max_entropy=1,):
        self.__max_entropy=max_entropy
        

    
class Bool:
    count=0
    def __init__(self,value=None):
        if value==None: self.value=None
        else:self.value=bool(value)
        self.__c=Bool.count
        Bool.count+=1
    @property
    def entropy(self):
        if self.value==None: return 1
        else: return 0
    @property
    def name(self):
        return f"Bool_{self.__c}"
    def __repr__(self):
        return f"{self.name} value={self.value} entropy={self.entropy}"
    
class Int:
    count=0
    def __init__(self, value=None,depth=32):
        self.depth=depth
        bounds=self.bounds
        if value==None:self.value=None
        elif bounds[0]<=value<=bounds[1]:self.value=int(value)
        else: raise ValueError(f"Invalid value in integer range {bounds}")
        self.__c=Int.count
        Int.count+=1
            
    @property
    def bounds(self):
        return -2**(self.depth-1),2**(self.depth-1)-1
    
    @property
    def entropy(self):
        if self.value==None: return self.depth
        else: return 0
    @property
    def name(self):
        return f"Int{self.depth}_{self.__c}"
    def __repr__(self):
        return f"{self.name} value={self.value} entropy={self.entropy}"

class UInt:
    count=0
    def __init__(self, value=None,depth=32):
        self.depth=depth
        if value==None:self.value=None
        elif 0<=value<2**self.depth:self.value=int(value)
        else: raise ValueError(f"Invalid value in unsigned integer range [0, {2**self.depth})")
        self.__c=UInt.count
        UInt.count+=1

    
    @property
    def entropy(self):
        if self.value==None: return self.depth
        else: return 0
    @property
    def name(self):
        return f"UInt{self.depth}_{self.__c}"
    def __repr__(self):
        return f"{self.name} value={self.value} entropy={self.entropy}"
  
class Float:
    count=0
    def __init__(self, value=None, exponent=8, mantissa=23 ):
        self.depth=1+mantissa+exponent
        self.__exponent=exponent
        self.__mantissa=mantissa
        self.__c=Float.count
        Float.count+=1
        
        inf=self.float_info
        if value==None:self.value=None
        elif -inf["max"]<=value<=inf["max"]:self.value=float(value)
        else:raise ValueError(f"Invalid value for {exponent}e{mantissa}m float{self.depth} range is [{inf['min']},{inf['max']}] ")
        
    @property
    def float_info(self):
        bias = 2**(self.__exponent - 1) - 1

        return {
                    "max_exp": 2**(self.__exponent-1),
                    "min_exp": 1 - 2**(self.__exponent-1),
                    "max": (2 - 2**(-self.__mantissa)) * 2**(2**(self.__exponent-1)-1),
                    "min": 2**(-bias+1),
                    "eps": 2**(-self.__mantissa),
                    "mant_dig": self.__mantissa,
                    "max_10_exp": int(2**(self.__exponent-1) * 0.301),
                    "min_10_exp": int((1 - 2**(self.__exponent-1)) * 0.301),
                    "bias": bias,
                    "radix": 2,
                    "rounds": 1,
                    "depth": self.depth,
                    "exp_dig": self.__exponent
                }

    @property
    def entropy(self):
        if self.value==None: return self.depth
        else: return 0
    @property
    def name(self):
        return f"Float{self.depth}({self.__exponent}e{self.__mantissa}m)_{self.__c}"
    def __repr__(self):
        return f"{self.name} value={self.value} entropy={self.entropy}"

class Qudit:
    count=0
    def __init__(self, value, num_levels=None,utol=1e-9):
        if type(value)==str:
            if value=="H":
                value=np.full(num_levels, 1/num_levels**.5)
            else:
                try:
                    a=int(value)
                    if 0<=a<num_levels:
                        value=np.zeros(num_levels)
                        value[a]=1.
                    else:raise ValueError(f"Qudit can't represent {a}, since # of levels is not sufficient")
                except:
                    raise ValueError("Invalid qudit initializer")
        if not abs(np.sum(np.abs(value)**2)-1)<=utol:
            raise ValueError("Qudit value is not unit vector")
        
        self.value=value
        self.__c=Qudit.count
        Qudit.count+=1
        
    @property
    def num_levels(self):return self.value.shape[0]
    @property
    def entropy(self):
        return 0
    @property
    def name(self):
        return f"{self.num_levels}-levelQudit_{self.__c}"
    def __repr__(self):
        return f"{self.name} value={self.value} entropy={self.entropy}"
    def __and__(self, other):return Qudit(np.kron(self.value, other.value))

class QuantumGate(object):
    count=0
    def __init__(self,data,utol=1e-8):
        if not isinstance(data, (np.ndarray,list, tuple)):
            raise TypeError("data should be an array")
        data=np.array(data)
        if len(data.shape)!=2 or data.shape[0]!=data.shape[1]:
            raise ValueError("data should be a square matrix")

        if not np.allclose(np.dot(data, data.conj().T), np.eye(data.shape[0]), utol):
            raise ValueError("data should be a unitary matrix. Either adjust data or utol(unitary tolerance)")
        self.__data=data
        self.__c=QuantumGate.count
        QuantumGate.count+=1
    
    def __repr__(self):
        return f"QuantumGate with {self.__data.shape[0]}-levels"        

    @property
    def data(self): return self.__data
    
    @property
    def value(self): return self.__data

    def set_data(self,data,utol): 
        if not isinstance(data, (np.ndarray,list, tuple)):
            raise TypeError("data should be an array")
        data=np.array(data)
        if len(data.shape)!=2 or data.shape[0]!=data.shape[1]:
            raise ValueError("data should be a square matrix")
        if not np.allclose(np.dot(data, data.conj().T), np.eye(data.shape[0]), utol):
            raise ValueError("data should be a unitary matrix. Either adjust data or utol(unitary tolerance)")
        self.__data=data

    @property
    def num_levels(self): return self.__data.shape[0]
    @property
    def entropy(self):return 0
    @property
    def name(self):return f"QuantumGate_{self.__c}"
    def __and__(self, other):return QuantumGate(np.kron(self.data, other.data))
    def __call__(self, sv): return Qudit(np.matmul(self.value, sv.value))

class NonHermitianGate:
    count=0
    def __init__(self, data):
        if isinstance(data,(list, tuple)): self.__data=np.array(data)
        elif isinstance(data, np.ndarray):self.__data=data
        else: raise TypeError("data must be a 2D array, list or tuple")
        if len(self.__data.shape)!=2:raise ValueError("data must be a 2D array, list or tuple")
        self.__c=NonHermitianGate.count
        NonHermitianGate.count+=1
    @property
    def value(self):return self.__data
    @property
    def name(self): return f"{self.__data[0]}×{self.__data[1]}NonHermetianGate{self.__c}"

    def __call__(self, sv):
        if sv.num_levels!=self.__data.shape[1]:raise ValueError("Dimesnions are unmatched for operating on NonHermetianGate")
        a=self.value @ sv.value
        a**=.5
        a/=(np.sum(np.abs(a)**2))**.5
        
        return Qudit(a)
    
    def __repr__(self):return f"{self.name}"
    
class Tensor(object):
    count=0
    def __init__(self, value, dtype=Float(0), shape=(1,)):
        if not isinstance(value, (np.ndarray,tuple, list)):
            if value==None:
                self.value=None
                if isinstance(shape, tuple):
                    self.shape=shape
                else:
                    raise TypeError("shape needs to be a tuple")
            else:
                raise TypeError("value needs to be an array")
        else:
            self.value=np.array(value)

        if not isinstance(dtype, (Bool, Int, UInt, Float, Qudit)):
            raise TypeError("dtye should be one of them (Bool, Int, UInt, Float, Qudit)")
        self.dtype=dtype
        self.__c=Tensor.count
        Tensor.count+=1
        self.predecessor=None

    @property
    def entropy(self):
        if type(self.value)!=np.ndarray:
            self.dtype.value=None
            return self.dtype.entropy*np.prod(self.shape)
        else:return 0

    def __repr__(self):
        if type(self.value)!=np.ndarray:
            return f"Tensor_{self.__c}: array of {repr(self.dtype)[:repr(self.dtype).index(' ')]} shape={self.shape} entropy={self.entropy}"
        else:
            return f"Tensor_{self.__c}: array of {repr(self.dtype)[:repr(self.dtype).index(' ')]} shape={self.value.shape} entropy={self.entropy}"
    @property
    def name(self): return f"Tensor_{self.__c}"
            
                
class Edge(object):
    count=0
    def __init__(self,sculk):
        if not isinstance(sculk,(Bool, Int, UInt, Float, Qudit, Tensor)):
            raise TypeError("sculk must be a Bool, Int, UInt, Float, Qudit or Tensor")
        self.sculk=sculk
        self.__c=Edge.count
        Edge.count+=1
        self.predecessor=None
        self.sv=None

    def vhdl(self):
        if type(self.sculk)==Bool:
            if self.sculk.value==None:
                return f"{self.sculk.name}:std_logic:='X'"
            elif self.sculk.value==0:
                return f"{self.sculk.name}:std_logic:='0'"
            elif self.sculk.value==1:
                return f"{self.sculk.name}:std_logic:='1'"
        elif type(self.sculk)==UInt:
            if self.sculk.value==None: return f"{self.sculk.name}:std_logic_vector({self.sculk.depth-1} downto 0):=\""+self.sculk.depth*"X"+"\""
            else:
                a=bin(self.sculk.value)[2:]
                return f"{self.sculk.name}:std_logic_vector({self.sculk.depth-1} downto 0):=\""+(self.sculk.depth-len(a))*"0"+a+"\""
        elif type(self.sculk)==Int:
            def twoscpl(x:int, maxn: int):
                a=bin(abs(x))[2:];   
                a="0"*(maxn-len(a))+a
                if x>=0:    
                    return a
                else:
                    b=""
                    for i in range(maxn):
                        if a[i]=="0":b+="1"
                        else: b+="0"
                    a=int(b,2)+1
                    a=bin(a)[2:];   
                    a="0"*(maxn-len(a))+a
                    return a
            if self.sculk.value==None: return f"{self.sculk.name}:std_logic_vector({self.sculk.depth-1} downto 0):=\""+self.sculk.depth*"X"+"\""
            else:
                a=twoscpl(self.sculk.value, self.sculk.depth)
                return f"{self.sculk.name}:std_logic_vector({self.sculk.depth-1} downto 0):=\""+a+"\""
                       
    @property
    def name(self): return f"Edge_{self.__c}"
    def __repr__(self):return f"{self.name} holding {self.sculk}"
    @property
    def entropy(self):return self.sculk.entropy
    def set_predecessor(self,node):self.predecessor=node
    def unpack(self,executor="np"):
        if self.sculk.value==None:
            self.predecessor.execute(executor)
        else:
            return self.sculk.value
    def q_info(self):
        if self.sculk.value==None: 
            if self.sv==None:self.predecessor.q_stream()
            return self.sv
        if type(self.sculk)==Bool:
            if self.sculk.value==False: return Qudit((1, 0))
            elif self.sculk.value==True: return Qudit((0,1))
            else: return Qudit((1, 0))
        elif type(self.sculk)==Qudit: return self.sculk
        elif type(self.sculk)==UInt: return Qudit(str(self.sculk.value), 2**self.sculk.depth)
        elif type(self.sculk)==Int: return Qudit(str(int(self.vhdl().split('"')[1],2)), 2**self.sculk.depth)
    
class Node(object):
    count=0
    def __init__(self, in_terminals, out_terminals, ops={"np":None, "cp":None, "tf":None, "vhdl":None,"qc":None}):
        if Edge in type(in_terminals).mro():
            in_terminals=(in_terminals,)
        elif isinstance(in_terminals,(tuple,list, np.ndarray)):
            for terminal in in_terminals:
                if not Edge in type(terminal).mro():
                    raise TypeError("each element of in_terminals must be an Edge or inherited from Edge")
        else: raise TypeError("in_terminals should be composed of Edge")
        
        if Edge in type(out_terminals).mro():
            out_terminals=(out_terminals,)
        elif isinstance(out_terminals,(tuple,list, np.ndarray)):
            for terminal in out_terminals:
                if not Edge in type(terminal).mro():
                    raise TypeError("each element of out_terminals must be an Edge or inherited from Edge")
        else: raise TypeError("out_terminals should be composed of Edge or inherited from Edge")
        
        self.__in_terminals=in_terminals
        self.__out_terminals=out_terminals
        self.__contemplate=None
        for terminal in out_terminals:
            terminal.set_predecessor(self)
        
        self.__qv_contemplate=None
        self.__c=Node.count
        Node.count+=1
        self.__ops=ops
        
    def execute(self,executor="np"):
        __=[i.unpack(executor) for i in self.__in_terminals]
        self.__contemplate=self.__ops[executor](*[i.unpack(executor) for i in self.__in_terminals])
        for i in range(len(self.__out_terminals)):
            self.__out_terminals[i].sculk.value=self.__contemplate[i]
    
    def edge_accumulator(self):
        edges=set([i.vhdl() for i in self.__in_terminals]+[i.vhdl() for i in self.__out_terminals]);
        inputs=set()        
        for i in self.__in_terminals:
            if i.predecessor!=None:
                edges=edges.union(i.predecessor.edge_accumulator())
        return edges

    def input_accumulator(self):
        inputs=set()
        for i in self.__in_terminals:
            if i.predecessor==None:
                """
                a=i.vhdl()
                b=a.index(":")
                c=a[b+1:].index(":")
                a=a[:b+1]+"in "+a[b+1:b+c+1]"""
                inputs.add(i.vhdl())
            else:
                inputs=inputs.union(i.predecessor.input_accumulator())
        return inputs

    def node_accumulator(self):
        operations=set()
        
        if self.__ops["vhdl_class"]==0:
            a=self.__ops["vhdl"](*[i.sculk.name for i in self.__in_terminals])
            for i in range(len(self.__out_terminals)):
                operations.add(f"{self.__out_terminals[i].sculk.name}<={a[i]};")
        elif self.__ops["vhdl_class"]==1:
            a=self.__in_terminals[0].sculk.depth
            for i in self.__in_terminals:
                if a!=i.sculk.depth: raise ValueError("Depth of {i} is unmatched with depth of {self.__in_terminals[0]}")
            for i in self.__out_terminals:
                if a!=i.sculk.depth: raise ValueError("Depth of {i} is unmatched with depth of {self.__in_terminals[0]}")
            operations.add(self.__ops["vhdl"](*([i.sculk.name for i in self.__in_terminals]+[i.sculk.name for i in self.__out_terminals]+[a])))
        for i in self.__in_terminals:
            if i.predecessor==None:
                pass
            else:
                operations=operations.union(i.predecessor.node_accumulator())
        return operations

    def transpile(self,target="vhdl"):
        if target.lower()=="vhdl":
            edges=self.edge_accumulator()
            inputs=self.input_accumulator()
            outputs=set([i.vhdl() for i in self.__out_terminals])
            signals=edges.difference(inputs).difference(outputs)
            nodes=self.node_accumulator()
            
            print("Edges",edges,"\nInputs",inputs, "\nSignals",signals,"\nNodes",nodes,"\nOutputs",outputs)
            text=vhdl_preset
            text+="library IEEE;\nuse IEEE.STD_LOGIC_1164.ALL;\nentity main is"
            text+="\nPort(\n\t"
            for input_edge in inputs:
                a=input_edge.index(":")
                b=input_edge[a+1:].index(":")
                text+=input_edge[:a]+":in "+input_edge[a+1:a+b+1]+";\n\t"
        
            for i,output_edge in enumerate(outputs):
                a=output_edge.index(":")
                b=output_edge[a+1:].index(":")
                text+=output_edge[:a]+":out "+output_edge[a+1:a+b+1]
        
                if i!=len(outputs)-1:
                    text+=";\n\t"
            text+="\n);\nend main;\narchitecture Behavioral of main is"
            for signal in signals:
                text+="\n\tsignal "+signal[:signal.rindex(":")]
                text+=";"
            text+="\nbegin"
            for node in nodes:
                text+="\n\t"+node
            text+="\nend architecture Behavioral;"
            return text

    def qv_contemplate(self):return self.__qv_contemplate
     
    def q_stream(self):
        if self.__qv_contemplate==None:
            self.__qv_contemplate=self.__ops["nhq"](*[i for i in self.__in_terminals])
        for i,edge in enumerate(self.__out_terminals):
            edge.sv=self.__qv_contemplate[i]
            
        
class LogicalAnd(Node):
    count=0
    def __init__(self, in_terminals, out_terminals):
        self.thermo=DigitalSoul.erg.ThermodynamicGate(2,8)
        super().__init__(in_terminals, out_terminals, ops={
            "np": lambda a,b:(np.logical_and(a,b),),
            "cp":lambda a,b:(cp.logical_and(a,b),),
            "tf":lambda a,b:(tf.logical_and(a,b),),
            "vhdl":lambda a,b: (f"{a} and {b}",),
            "vhdl_class":0,
            "nhq": lambda a,b: (NonHermitianGate(((1.,1,1,0),(0,0,0,1)))(a.q_info()&b.q_info()),)
        })
        self.__c=LogicalAnd.count
        LogicalAnd.count+=1
        
class LogicalOr(Node):
    count=0
    def __init__(self, in_terminals, out_terminals):
        
        self.thermo=DigitalSoul.erg.ThermodynamicGate(2,14)
        super().__init__(in_terminals, out_terminals, ops={
            "np": lambda a,b:(np.logical_or(a,b),),
            "cp":lambda a,b:(cp.logical_or(a,b),),
            "tf":lambda a,b:(tf.logical_or(a,b),),
            "vhdl":lambda a,b: (f"{a} or {b}",),
            "vhdl_class":0,
            "nhq": lambda a,b: (NonHermitianGate(((1.,0,0,0),(0,1,1,1)))(a.q_info()&b.q_info()),)
        })
        self.__c=LogicalOr.count
        LogicalOr.count+=1


class LogicalXor(Node):
    count=0
    def __init__(self, in_terminals, out_terminals):
        self.thermo=DigitalSoul.erg.ThermodynamicGate(2,6)
        super().__init__(in_terminals, out_terminals, ops={
            "np": lambda a,b:(np.logical_xor(a,b),),
            "cp":lambda a,b:(cp.logical_xor(a,b),),
            "tf":lambda a,b:(tf.logical_xor(a,b),),
            "vhdl":lambda a,b: (f"{a} xor {b}",),
            "vhdl_class":0,
            "nhq": lambda a,b: (NonHermitianGate(((1.,0,0,1.),(0,1,1,0)))(a.q_info()&b.q_info()),)
        })
        self.__c=LogicalXor.count
        LogicalXor.count+=1
        
class LogicalNot(Node):
    count=0
    def __init__(self, in_terminals, out_terminals):
        self.thermo=DigitalSoul.erg.ThermodynamicGate(1,1)
        super().__init__(in_terminals, out_terminals, ops={
            "np": lambda a:(np.logical_not(a),),
            "cp":lambda a:(cp.logical_not(a),),
            "tf":lambda a:(tf.logical_not(a),),
            "vhdl":lambda a: (f"not({a})",),
            "vhdl_class":0,
            "nhq": lambda a,b: (QN.x)(a.q_info(),)
        })
        self.__c=LogicalNot.count
        LogicalNot.count+=1
        
class ScalarAdd(Node):
    count=0
    def __init__(self,in_terminals, out_terminals):
        self.__c=ScalarAdd.count;ScalarAdd.count+=1
        super().__init__(in_terminals, out_terminals, ops={
            "np":lambda a,b: (np.add(a,b),),
            "cp":lambda a,b: (cp.add(a,b),),
            "tf":lambda a,b: (tf.add(a,b),),
            "vhdl": lambda a,b,c,N:"adder_"+str(self.__c)+f": entity work.FixedPointAdder generic map(N=>{N}) port map(a=>{a}, b=>{b}, overflow=>open, c=>{c});",
            "vhdl_class":1
            })

class ScalarSubtract(Node):
    count=0
    def __init__(self,in_terminals, out_terminals):
        self.__c=ScalarSubtract.count;ScalarAdd.count+=1
        super().__init__(in_terminals, out_terminals, ops={
            "np":lambda a,b: (np.subtract(a,b),),
            "cp":lambda a,b: (cp.subtract(a,b),),
            "tf":lambda a,b: (tf.subtract(a,b),),
            "vhdl": lambda a,b,c,N:"adder_"+str(self.__c)+f": entity work.FixedPointSubtractor generic map(N=>{N}) port map(a=>{a}, b=>{b}, overflow=>open, c=>{c});",
            "vhdl_class":1
            })
        
class QN:
    i=QuantumGate([[1,0],[0,1]])
    x=QuantumGate([[0,1],[1,0]])
    y=QuantumGate([[0,-1j],[1j,0]])
    z=QuantumGate([[1,0],[0,-1]])
    h=QuantumGate([[1/2**.5,1/2**.5],[1/2**.5,-1/2**.5]])
    cx=i&x
    ccx=QuantumGate([[1,0,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0,0],
                     [0,0,1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0],
                     [0,0,0,0,1,0,0,0],
                     [0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,1,0]])
    
    
vhdl_preset=""
"""library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity HalfAdder is Port(a,b:in std_logic; c_out, sum:out std_logic); end HalfAdder;

architecture Behavioral of HalfAdder is
begin
    c_out<=a and b;
    sum<=a xor b;
end Behavioral;
------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity FullAdder is
    Port ( a : in STD_LOGIC;
           b : in STD_LOGIC;
           c_in : in STD_LOGIC;
           c_out : out STD_LOGIC;
           sum : out STD_LOGIC);
end FullAdder;

architecture Behavioral of FullAdder is

begin
    sum<=a xor b xor c_in;
    c_out<=((a xor b) and c_in) or (a and b);
end Behavioral;

------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity FixedPointAdder is -- a+b=c
    generic(N:integer:=32);
    port(a,b:in std_logic_vector(N-1 downto 0); c: out std_logic_vector(N-1 downto 0); overflow:out std_logic);
end FixedPointAdder;

architecture Behavioral of FixedPointAdder is
    signal c_inter: std_logic_vector(N-1 downto 0);
begin
    FA1: entity work.FullAdder port map(a=>a(0), b=>b(0), c_in=>'0', c_out=>c_inter(0), sum=>c(0));
    FA_array: for i in 1 to N-1 generate FA: entity work.FullAdder port map(a=>a(i), b=>b(i),c_in=>c_inter(i-1),c_out=>c_inter(i), sum=>c(i));
    end generate FA_array;
    overflow<=c_inter(N-1);

end architecture Behavioral;
-----------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity FixedPointSubtractor is -- a - b
    generic(N: integer:=32);
    port(a,b:in std_logic_vector(N-1 downto 0); c: out std_logic_vector(N-1 downto 0); overflow: out  std_logic);
end FixedPointSubtractor;

architecture Behavioral of FixedPointSubtractor is
    signal c_inter,b_inter: std_logic_vector(N-1 downto 0);
begin
    b_inter<=not(b);
    FA1: entity work.FullAdder port map(a=>a(0), b=>b_inter(0),c_in=>'1',c_out=>c_inter(0), sum=>c(0));
    FA_array: for i in 1 to N-1 generate FA: entity work.FullAdder port map(a=>a(i), b=>b_inter(i),c_in=>c_inter(i-1),c_out=>c_inter(i), sum=>c(i)); end generate FA_array;
    overflow<=not c_inter(N-1);
end Behavioral;
----------------------------------------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity FixedPointMultiplier_MixedPrecision is
    generic(N:integer:=128);
    port(a,b:in std_logic_vector(N-1 downto 0); c: out std_logic_vector(2*N-1 downto 0));
end FixedPointMultiplier_MixedPrecision;

architecture Behavioral of FixedPointMultiplier_MixedPrecision is
    signal anded: std_logic_vector(N**2-1 downto 0);
    signal sumout: std_logic_vector(N**2+N-1 downto 0);--N rows, N+1 cols
    signal carries: std_logic_vector(N-1 downto 0);
begin
    and_array: for i in 0 to N-1 generate 
        inner_loop: for j in 0 to N-1 generate
            gated: anded(N*i+j)<=a(j) and b(i); 
        end generate inner_loop; 
    end generate and_array;
    carries(0)<='0';
    sumout(N downto 1)<=anded(N-1 downto 0);

    add_array: for i in 1 to N-1 generate
        adder: entity work.FixedPointAdder generic map(N=>N+1) port map(a=>carries(i-1)&sumout((N+1)*(i)-1 downto (N+1)*(i-1)+1),
                                                                        b=>anded(N*(i+1)-1 downto N*i)&'0',
                                                                        overflow=>carries(i),
                                                                        c=>sumout((N+1)*(i+1)-1 downto (N+1)*i) );
    end generate add_array;

    outport1: for i in 0 to N-2 generate
        outport: c(i)<=sumout((N+1)*(i)+1);
    end generate outport1;

    c(2*N-2 downto N-1)<=sumout(N**2+N-1 downto N**2);
    c(2*N-1)<=carries(N-1);
end Behavioral;
----------------------------------------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity FixedPointFMA is
    generic(N:natural:=32);
    port(mul1, mul2, add: in std_logic_vector(N-1 downto 0); fused: out std_logic_vector(N-1 downto 0); overflow: out std_logic);
end FixedPointFMA;


architecture Behavioral of FixedPointFMA is
    signal distillate: std_logic_vector(2*N-1 downto 0);
    signal of_mul, of_add:std_logic;
begin


    multiplier: entity work.FixedPointMultiplier_MixedPrecision generic map(N=>N) port map(a=>mul1, b=>mul2, c=>distillate);
    process(distillate) begin
    if distillate(2*N-1 downto N) = (N-1 downto 0 => '0') then 
        of_mul<='0'; 
    else of_mul <= '1'; 
        end if;
    end process;
    adder: entity work.FixedPointAdder generic map(N=>N) port map(a=>distillate(N-1 downto 0), b=>add, c=>fused, overflow=>of_add);
    
    overflow<=of_mul or of_add;
end architecture Behavioral;"""
