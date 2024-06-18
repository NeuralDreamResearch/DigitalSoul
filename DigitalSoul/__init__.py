# DigitalSoul: Unified platform for CPU, GPU, FPGA, Quantum Computing
print("Interacting with reality...")
import numpy as np
import DigitalSoul.erg # Import Thermodynamic assets
import DigitalSoul.dscpp # Import C++ assets
import graphviz


version="1.2-dev"
UUID="00008"
tf_available=False

print(f"Dev UUID : {UUID}")
try:
    #import tensorflow as tf
    print("Tensorflow is available")
    tf_available=True
    tf=np
except:
    print("Tensorflow is skipped")
    tf=np    
try:
    import cupy as cp
    print("GPU resources are available. Eager executor can access it.")

except:
    print("GPU resources are not available")
    cp=np

residency=DigitalSoul.dscpp.residency
dtype=DigitalSoul.dscpp.dtype

Bool=dtype(0)
UInt8=dtype(1,(8,))
UInt16=dtype(1,(8,))
UInt32=dtype(1,(8,))
UInt64=dtype(1,(8,))
UInt=UInt32

Int8=dtype(2,(8,))
Int16=dtype(2,(8,))
Int32=dtype(2,(8,))
Int64=dtype(2,(8,))
Int=Int32

Float16 = dtype(3, (5, 10))
Float32 = dtype(3, (8, 23))
Float64 = dtype(3, (11, 52))
Float=Float32


namespace=set()

        

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
    


    
    
    
class Edge(DigitalSoul.dscpp.residency):


    def __init__(self, value, shape=None,dtype_=Float,residency=residency(), name=None):
        
        
        
        if name==None:
        
            self.name="DS_UUID_"+str(id(self))
        elif not isinstance(name, str):
            raise TypeError("name should be a string")
        elif name in namespace:
            raise ValueError("name should not be in namespace. Probably used previously.")
        elif name.startswith("DS_UUID_"):
            raise ValueError("name cannot start with DS_UUID_")
        else:
            self.name=name
        namespace.add(self.name)
        
        if not isinstance(dtype_, dtype):
            raise TypeError("dtype should be a dtype class of Digital Soul")


        self.__dtype=dtype_
        
        if not(shape==None or isinstance(shape, (tuple, list,np.ndarray))):
            raise TypeError("shape should be None or array")
        elif shape==None:
            self.__shape=None
        else:
            self.__shape=tuple(shape)
        
        if residency==None:super().__init__(*DigitalSoul.dscpp.residency.toBoolArray())
            
        elif isinstance(residency, DigitalSoul.dscpp.residency): super().__init__(*residency.toBoolArray())
        
        self.sculk={}
        if self.isCPUAvailable():
            if not( value is None):
                self.sculk["np"]=np.array(value)
            else:
                self.sculk["np"]=None
        if self.isGPUAvailable():
            if value!=None:
                self.sculk["cp"]=cp.array(value)
            else:
                self.sculk["cp"]=None

        if self.isQNetworkAvailable():
            self.sculk["qn"]=None
            
        
    
        self.pre=None
        self.post=None
    @property
    def dtype(self):return self.__dtype
    
    @property
    def shape(self):return self.__shape


    def __del__(self):
        namespace.remove(self.name)


    def rename(self, new_name=None):
        namespace.remove(self.name)
        
        
        if new_name==None: 
            self.name="DS_UUID_"+str(id(self))
        elif not isinstance(new_name, str):
            raise TypeError("name should be a string")
        elif new_name in namespace:
            raise ValueError("name should not be in namespace. Probably used previously.")
        elif new_name.startswith("DS_UUID_"):
            raise ValueError("name cannot start with DS_UUID_")
        else:
            self.name=new_name
        
        namespace.add(new_name)
    
    def register_graph(self, pre, post):
        self.pre=pre
        self.post=post
    
    def register_pre(self,pre):
        self.pre=pre
        
    def register_post(self, post):
        self.post=post
        
    def compute_request(self, backend):
        if self.sculk[backend] is None:
            self.pre.execute(backend)
    
    @property
    def entropy(self):
        isNone=True
        for arr in self.sculk.values():
            if arr!=None: isNone=False
            
        if isNone:
            if self.shape==None:
                return self.dtype.get_num_bits()
            else:
                return self.dtype.get_num_bits()*np.prod(self.shape)
        else:
            return 0

class Node(DigitalSoul.dscpp.residency):     
    def __init__(self, in_terminals, out_terminals, ops={"np":None, "cp":None, "tf":None, "vhdl":None,"qc":None},residency=residency(1,signal=1),name=None):
        
        if residency==None:super().__init__(*DigitalSoul.dscpp.residency.toBoolArray())
            
        elif isinstance(residency, DigitalSoul.dscpp.residency): super().__init__(*residency.toBoolArray())
        
        for terminal in in_terminals:
            if Edge not in type(terminal).mro():
                raise TypeError("in_terminals should be an array of Edges")
            if not self.toBoolArray()== terminal.toBoolArray():
                raise ValueError("both Edge and Node should have same residency")
            terminal.register_post(self)
        
        for terminal in out_terminals:
            if Edge not in type(terminal).mro():
                raise TypeError("out_terminals should be an array of Edges")
            terminal.register_pre(self)
            
        self.__in_terminals=in_terminals
        self.__out_terminals=out_terminals
        if name==None:
            self.name="DS_UUID_"+str(id(self))
        elif not isinstance(name, str):
            raise TypeError("name should be a string")
        elif name in namespace:
            raise ValueError("name should not be in namespace. Probably used previously.")
        elif name.startswith("DS_UUID_"):
            raise ValueError("name cannot start with DS_UUID_")
        else:
            self.name=name
        namespace.add(self.name)
        
        
        if residency==None:super().__init__(*DigitalSoul.dscpp.residency.toBoolArray())
            
        elif isinstance(residency, DigitalSoul.dscpp.residency): super().__init__(*residency.toBoolArray())
        
        self.__ops=ops


        
    def __del__(self):
        namespace.remove(self.name)
            
    def rename(self, new_name=None):
        namespace.remove(self.name)
        if new_name==None: 
            self.name="DS_UUID_"+str(id(self))
        elif not isinstance(new_name, str):
            raise TypeError("name should be a string")
        elif new_name in namespace:
            raise ValueError("name should not be in namespace. Probably used previously.")
        elif new_name.startswith("DS_UUID_"):
            raise ValueError("name cannot start with DS_UUID_")
        else:
            self.name=new_name
        
        namespace.add(new_name)
        
    def execute(self, backend="np"):
        for pre in self.__in_terminals:
            pre.compute_request(backend)

        args = []

        for pre in self.__in_terminals:
            args.append(pre.sculk[backend])
        

        result=self.__ops[backend](*args)
        for i in range(len(self.__out_terminals)):
            self.__out_terminals[i].sculk[backend]=result[i]
            
    def vhdl_descriptor(self):
        code=""
        if self.__ops["vhdl_class"]==0:
            args = []

            for pre in self.__in_terminals:
                args.append(pre.name)

            for i in range(len(self.__out_terminals)):
                code+=f"\t{self.__out_terminals[i].name} <= {self.__ops['vhdl'](*args)[i]};\n"
                
            return code
        
    @property
    def in_terminals(self):
        return self.__in_terminals
        
    @property
    def out_terminals(self):
        return self.__out_terminals        

class ProductionNode(object):
    count=0
    def __init__(self,  ops={"np":None, "cp":None, "tf":None, "vhdl":None,"qc":None},residency=residency(1, signal=True),super_name=None):
        self.__ops=ops
        self.__super_name=super_name
        self.residency=residency
    
    def __call__(self, in_terminals, out_terminals):
        ProductionNode.count+=1
        return Node(in_terminals,out_terminals,self.__ops,self.residency,self.__super_name+str(ProductionNode.count-1))
        
class ComputationalGraph(object):
    def __init__(self,nodes,edges=None,name=None):
        self.nodes=nodes
        self.edges=None
        if not(edges is None):
            self.edges=edges
        else:
            self.edges=set()
            for node in self.nodes:
                for edge in node.in_terminals:
                    self.edges.add(edge)
                for edge in node.out_terminals:
                    self.edges.add(edge)

        self.starts=[]
        self.ends=[]
        self.graphiz_graph= graphviz.Digraph(comment='Computational Graph')
        self.graphify()
        self.name=str(name)
        
    def graphify(self):
        for i in self.nodes:
            self.graphiz_graph.node(i.name, i.name)
        for j in self.edges:
            if j.pre!=None and j.post!=None:
                self.graphiz_graph.edge(j.pre.name, j.post.name, j.name)
            elif j.pre==None and j.post!=None:
                self.starts.append(j)
                self.graphiz_graph.node(j.name, j.name)
                self.graphiz_graph.edge(j.name, j.post.name, j.name)
            elif j.pre!=None and j.post==None:
                self.ends.append(j)
                self.graphiz_graph.node(j.name, j.name)
                self.graphiz_graph.edge(j.pre.name, j.name, j.name)
                
    def memory_layout(self):
        num_bits=[0,0,0] # BRAM, Reg, Signal
        for edge in self.edges:
            if edge.isBRAMAvailable():
                if edge.shape==None:
                    num_bits[0]+=edge.get_num_bits()
                else:
                    num_bits[0]+=np.prod(edge.shape)*edge.get_num_bits()
                    
            elif edge.isRegisterAvailable():
                if edge.shape==None:
                    num_bits[1]+=edge.get_num_bits()
                else:
                    num_bits[1]+=np.prod(edge.shape)*edge.get_num_bits()
            
            elif edge.isSignalAvailable():
                if edge.shape==None:
                    num_bits[2]+=edge.get_num_bits()
                else:
                    num_bits[2]+=np.prod(edge.shape)*edge.get_num_bits()

        return num_bits
      
    def render(self,view):
        self.graphiz_graph.render('example_graph', format='svg', view=view)
        
    def execute(self,backend):
        for end in self.ends:
            end.compute_request(backend)
        
    def generate(self,target="VHDL"):
        if target.upper()=="VHDL":
            code="""library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;\n"""
            code+="entity "+self.name+" is\nPort(\n"
            for interm in self.starts:
                code+=f"\t{interm.name}: in {interm.dtype.vhdl_descriptor()};\n"
            for outterm in self.ends[:-1]:
                code+=f"\t{outterm.name}: out {outterm.dtype.vhdl_descriptor()};\n"
            code+=f"\t{self.ends[-1].name}: out {self.ends[-1].dtype.vhdl_descriptor()}\n"
            code+=f");\nend {self.name};\narchitecture Behavioral of {self.name} is\n"
            for inter_node in set(self.edges)-set(self.starts)-set(self.ends):
                if inter_node.isSignalAvailable():
                    code+=f"\tsignal {inter_node.name}:{inter_node.dtype.vhdl_descriptor()};\n"
            code+="begin\n"
            for node in self.nodes:
                code+=node.vhdl_descriptor()
            code+="end architecture Behavioral;"
            return code
