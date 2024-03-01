# DigitalSoul: Unified platform for CPU, GPU, FPGA, Quantum Computing
print("Interacting with reality...")
import numpy as np
import re
import sys
import math
#import networkx as nxp
from scipy.special import expn
try:
    import tensorflow as tf
    print("Tensorflow is available")
except:
    print("Tensorflow is skipped")
    tf=np
try:
    import qiskit
    print("Classical&Quantum Resources are available. Eager executor can access both.")
    try:
        import QALU
    except:
        print("Install QALU for accessing Quantum Resources. Install with:\npip install QALU")
except:
    print("Classical Resources are available")
    
try:
    import cupy as cp
    print("GPU resources are available. Eager executor can access it.")

except:
    print("GPU resources are not available")
    cp=np
class VHDL_Factory(object):
    preset="""
    ----------------------------------------------------------------------------------
    -- Company: Neural Dream Reseach
    -- Engineer: Ali Hakim Taşkıran
    -- 
    -- Create Date: 01/05/2024 09:24:57 PM
    -- Design Name: Arithmetic Units
    -- Module Name: Add-Subtract-Multiply
    -- Project Name: Largon Accelerated Processing Units
    -- Target Devices: 
    -- Tool Versions: 
    -- Description: 
    -- 
    -- Dependencies: 
    -- 
    -- Revision:
    -- Revision 0.01 - File Created
    -- Additional Comments:
    -- 
    ----------------------------------------------------------------------------------


    library IEEE;
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
    end architecture Behavioral;
    """

class dtype(object):
    def __init__(self,name=None,cpu=None,gpu=None,qubit=None,vhdl=None,matlab=None,convertables=None,entropy=None):
        self.dname=name
        self.cpu_compat=cpu
        self.gpu_compat=gpu
        self.qubit_compat=qubit
        self.vhdl_compat=vhdl
        self.matlab_compat=matlab
        self.convertables=convertables
        self.entropy=entropy
    
    @property
    def dtype(self):
        return self.dname


class Int(dtype):
    count=0
    apriori_count=0
    namespace=set()
    def __init__(self,value=None,depth=32,name=None):
        
        if not(value==None or isinstance(value, (int, np.integer))):raise TypeError(f"{value} is not a valid Int.")
        if value!=None and not 2**(depth-1)-1>value>-2**(depth-1):raise ValueError(f"Int with depth {depth} cannot handle the value {value}. Consider to increase depth or decrease the value")
        
        if name in Int.namespace:raise ValueError(f"name={name} of the Int is used previously. Pick another name")
        if not (name==None or type(name)==str):raise TypeError("name of the Int is either a string or None")
        if name:
            self.__name=name
            Int.count+=1
            
        else:
            Int.count+=1
            Int.apriori_count+=1
            self.__name="Int"+str(depth)+"_"+str(Int.apriori_count)
            
        Int.namespace.add(self.__name)
        
        self.__value=value
        self.__depth=depth
        if depth>64:
            super().__init__(f"Int{depth}",True,False,True,True,True,entropy=depth)
        else:
            super().__init__(f"Int{depth}",True,True,True,True,True,entropy=depth)

    
    def __repr__(self):
        return f"Int{self.__depth} {self.__value}, {self.__name}"
    
    @property
    def value(self):
        return self.__value
    
    @property
    def depth(self):
        return self.__depth
    
    def set_value(self,val):
        self.__value=val
        
    @property
    def name(self):
        return self.__name
    
    @property 
    def int_info(self):
        return {"max":2**(self.__depth-1)-1, "min":-2**(self.__depth-1)}
    

    def check_value_is_appropriate(self, value):
        if not (value == None or isinstance(value,(int,np.integer))):
            raise TypeError(f"{value} is not a valid Int.")
        if value is not None and not (2**(self.__depth-1) - 1 > value > -2**(self.__depth-1)):
            raise ValueError(f"Int with depth {self.__depth} cannot handle the value {value}. Consider increasing depth or decreasing the value.")
        return True

    
class Bool(dtype):
    count=0
    apriori_count=0
    namespace=set()
    def __init__(self, value,name=None):
        if not(value==None or type(value)==bool):raise TypeError(f"{value} is not a valid Bool.")
        
        if name in Bool.namespace:raise ValueError(f"name={name} of the Bool is used previously. Pick another name")
        if not (name==None or type(name)==str):raise TypeError("name of the Bool is either a string or None")
        if name:
            self.__name=name
            Bool.count+=1
            
        else:
            Bool.count+=1
            Bool.apriori_count+=1
            self.__name="Bool_"+str(Bool.apriori_count)
            
        Bool.namespace.add(self.__name)
        
        self.__value=value
        
        super().__init__("Bool",True,True,True,True,True,{None},1)
        
    def __repr__(self):
        return f"Bool {self.__value}, {self.__name}"
    
    @property
    def value(self):
        return self.__value
    
    def set_value(self,val):
        self.__value=val
        
    @property
    def name(self):
        return self.__name
    
    def check_value_is_appropriate(self, value):
        if not (value is None or isinstance(value, bool)):
            raise TypeError(f"{value} is not a valid Bool.")
        return True
    
class BoolArray(dtype):
    count=0
    apriori_count=0
    namespace=set()
    def __init__(self,value, name=None):
        if type(value)==str:
            if not re.match("^[01]+$",value): raise ValueError("Bool arrays only take 0's and 1's")
            self.__value = np.array([True if char == "1" else False for char in value])



        elif type(value) in {list, np.ndarray}:
            for i in range(len(value)):
                if value[i] not in {True, False, "0","1",0,1}:
                    raise ValueError("BoolArray can only be array of 1, 0, True and False")
                if value[i] in {"0","1",0,1}: 
                    value[i]=True if value[i] else False
            self.__value=np.array(value)
            
        elif type(value)==tuple:
            value=list(value)
            for i in range(len(value)):
                if value[i] not in {True, False, "0","1",0,1}:
                    raise ValueError("BoolArray can only be array of 1, 0, True and False")
                if value[i] in {"0","1",0,1}: 
                    value[i]=True if value[i] else False
            self.__value=np.array(value)
            
        elif value==None:
            self.__value=value
        
        if name in BoolArray.namespace:raise ValueError(f"name={name} of the BoolArray is used previously. Pick another name")
        if not (name==None or type(name)==str):raise TypeError("name of the BoolArray is either a string or None")
        if name:
            self.__name=name
            BoolArray.count+=1
            
        else:
            BoolArray.count+=1
            BoolArray.apriori_count+=1
            self.__name="BoolArray_"+str(BoolArray.apriori_count)
            
        BoolArray.namespace.add(self.__name)
        
        
        try:
            super().__init__("BoolArray",True,True,True,True,True,{None},len(value))
        except TypeError:
            super().__init__("BoolArray",True,True,True,True,True,{None},None)
    
    @property    
    def value(self):
        return self.__value
    
    def set_value(self,val):
        self.__value=val
        self.entropy=len(val)
        
    @property
    def name(self):
        return self.__name
    
    def __repr__(self):
        return f"BoolArray {self.__value}, {self.__name}"
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__value[index]
        raise IndexError("Unsupported indexing")
        
    def __setitem__(self, index,values):
        if isinstance(index, slice):
            self.__value[index]=[True if val in {True, 1, "1"} else False for val in values]
        

    def check_value_is_appropriate(self, value):
        if isinstance(value, str):
            if not re.match("^[01]+$", value):
                raise ValueError("Bool arrays only take 0's and 1's.")
        elif isinstance(value, (list, np.ndarray)):
            for item in value:
                if item not in {True, False, "0", "1", 0, 1}:
                    raise ValueError("BoolArray can only be an array of 1, 0, True, and False.")
        elif isinstance(value, tuple):
            for item in value:
                if item not in {True, False, "0", "1", 0, 1}:
                    raise ValueError("BoolArray can only be an array of 1, 0, True, and False.")
        elif value is not None:
            raise TypeError(f"{value} is not a valid BoolArray.")
        return True

        
class Qubit(dtype):
    count=0
    apriori_count=0
    namespace=set()
    def __init__(self,value,name=None,normalizer_error=1e-10, entropy_disturber=1e-20):
        
        if not isinstance(normalizer_error,float):
            raise TypeError("normalizer_error must be a float")
            
        if not isinstance(entropy_disturber,float):
            raise TypeError("entropy_disturber must be a float")
            
        if not entropy_disturber>0:
            raise ValueError("entropy_disturber must be a small positive number")
            
        if not normalizer_error>0:
            raise ValueError("normalizer_error must be a small positive number")
        
        if len(value)!=2: raise ValueError("Qubit system can only include 2 components")
        
        sums=0
        for e in value:
            if not isinstance(e,(int, float,complex,np.floating,np.integer,np.complexfloating)):
                raise TypeError("Qubit can only hold statevector composed of complex, float or integer")
            sums+=abs(e)**2
        
        if abs(sums-1)>normalizer_error:
            raise ValueError("state of Qubit must be normalized to 1")

        
        if name in Qubit.namespace:raise ValueError(f"name={name} of the Qubit is used previously. Pick another name")
        if not (name==None or type(name)==str):raise TypeError("name of the Qubit is either a string or None")
        if name:
            self.__name=name
            Qubit.count+=1
            
        else:
            Qubit.apriori_count+=1
            Qubit.count+=1
            self.__name="Qubit_"+str(Qubit.apriori_count)
            
        Qubit.namespace.add(self.__name)
        
        
        self.__value=np.array(value)
        entropy=0
        for e in np.round(np.linalg.eig(self.density_matrix)[0]/entropy_disturber)*entropy_disturber:
            if not abs(e)<entropy_disturber:
                entropy-=e*math.log(e,2)

        super().__init__("Qubit",True,True,True,True,True,{None},entropy=entropy)
    
    @property
    def density_matrix(self):
        return np.outer(self.__value,self.__value)
    
    @property    
    def value(self):
        return self.__value
    
    def set_value(self,val):
        self.__value=val
        
    @property
    def name(self):
        return self.__name
    
    def __repr__(self):
        return f"Qubit {self.__value}, {self.__name}"
    

    def check_value_is_appropriate(self, value):
        if len(value) != 2:
            raise ValueError("Qubit system can only include 2 components")
        sums = 0
        for e in value:
            if not isinstance(e, (int, float, complex, np.floating, np.integer, np.complexfloating)):
                raise TypeError("Qubit can only hold a statevector composed of complex, float, or integer values")
            sums += abs(e)**2
        if abs(sums-1) > self.__normalizer_error:
            raise ValueError("State of Qubit must be normalized to 1")
        return True

    
class Qudit(dtype):
    count=0
    apriori_count=0
    namespace=set()
    def __init__(self,value,N=2,name=None,normalizer_error=1e-8, entropy_disturber=1e-12):
        
        
        if type(N)!=int:
            raise TypeError("N must be an integer")     

        if(isinstance(value,str)):
            if value=="0":
                value=np.zeros((N,))
                value[0]=1.
            elif value=="H":
                value=np.full((N,),(1/N)**.5)
            else:
                try:
                    a=int(value)
                    if a>=N or a<0:
                        raise ValueError("Numeral state descriptor is invalid")
                    value=np.zeros((N,))
                    value[a]=1.
                except:   
                    raise ValueError("Invalid initial state description")
        elif(isinstance(value, Int)):
            N=value.depth
            a=value.value
            value=np.zeros((2**N,))
            value[a]=1.

        elif len(value)!=N:
            raise ValueError("{N}-Dimensional Qudit must include {N} components")
        
        self.__N=N
        
        if not isinstance(normalizer_error,float):
            raise TypeError("normalizer_error must be a float")
            
        if not isinstance(entropy_disturber,float):
            raise TypeError("entropy_disturber must be a float")
            
        if not entropy_disturber>0:
            raise ValueError("entropy_disturber must be a small positive number")
            
        if not normalizer_error>0:
            raise ValueError("normalizer_error must be a small positive number")
        
        for i in range(N):
            if not isinstance(value[i], (int, float,complex,np.floating,np.integer,np.complexfloating)):
                raise TypeError(f"{N}-dimensional Qudit can only hold statevector composed of complex, float or integer")
        
        if abs(np.sum(np.abs(value)**2)-1)>normalizer_error:
            raise ValueError(f"state of {N}-dimensional Qudit must be normalized to 1")
        
        
        if name in Qudit.namespace:raise ValueError(f"name={name} of the Qudit is used previously. Pick another name")
        if not (name==None or type(name)==str):raise TypeError("name of the Qudit is either a string or None")
        if name:
            self.__name=name
            Qudit.count+=1
            
        else:
            Qudit.count+=1
            Qudit.apriori_count+=1
            self.__name="Qudit_"+str(Qudit.apriori_count)

        Qudit.namespace.add(self.__name)
        self.__value=np.array(value)
        
        entropy=0
        for e in np.round(np.linalg.eig(self.density_matrix)[0]/entropy_disturber)*entropy_disturber:
            if not abs(e)<entropy_disturber:
                entropy-=e*np.math.log(e,2)
        super().__init__(f"Qudit{N}",True,True,True,True,True,{None},entropy=entropy)
    
    @property    
    def value(self):
        return self.__value
    
    def set_value(self,val):
        self.__value=val
        
    @property
    def name(self):
        return self.__name
    
    @property
    def density_matrix(self):
        return np.outer(self.__value,self.__value)
    
    def __repr__(self):
        return f"{self.__N}-Dimensional Qudit {self.__value}, {self.__name}"
    
    def check_value_is_appropriate(self, value):
        
        if len(value) != self.__N:
            raise ValueError(f"{self.__N}-Dimensional Qudit must include {self.__N} components")
        sums = 0
        for e in value:
            if not isinstance(e, (int, float, complex, np.floating, np.integer, np.complexfloating)):
                raise TypeError(f"{self.__N}-dimensional Qudit can only hold a statevector composed of complex, float, or integer values")
            sums += abs(e)**2
        if abs(sums-1) > self.__normalizer_error:
            raise ValueError(f"State of {self.__N}-dimensional Qudit must be normalized to 1")
        return True
    
    @property
    def num_levels(self):
        return self.__N
   
class NontangledQudit(dtype):
    count=0
    apriori_count=0
    namespace=set()
    def __init__(self, qudit_cluster,name=None):
        
        #Implement here class assets
        if(not isinstance(qudit_cluster, dict)):
            raise TypeError("qudit_cluser must be a dictionary, where keys are positions and values must be qudits correspond that position")
        
    
        is_start_descriptor=True
        descriptor_type=None

        N=0
        for descriptor in qudit_cluster:
            if not is_start_descriptor:
                if descriptor_type!=type(descriptor):
                    raise ValueError("Invalid  Descriptor Type in UntangledQubits")
            else:
                descriptor_type=type(descriptor)
                is_start_descriptor=False
            
            if descriptor_type==int:
                N+=qudit_cluster[descriptor].num_levels

            elif descriptor_type==tuple:
                N+=len(descriptor)

        check_array=np.full(N,False, dtype=bool)

        if descriptor_type==int:
            for descriptor in qudit_cluster:
                if N>descriptor>=0:
                    for i in range(0, qudit_cluster[descriptor].num_levels):
                        if check_array[descriptor+i]==False:
                            check_array[descriptor+i]=True
                        else:
                            raise ValueError(f"Qudit cluster has multiple values in index-{descriptor+i}")
                else:
                    raise ValueError(f"Invalid descriptor {descriptor})
        if name in NontangledQudit.namespace:raise ValueError(f"name={name} of the NontangledQudit is used previously. Pick another name")
        if not (name==None or type(name)==str):raise TypeError("name of the NontangledQudit is either a string or None")
        if name:
            self.__name=name
            NontangledQudit.count+=1
            
        else:
            NontangledQudit.count+=1
            NontangledQudit.apriori_count+=1
            self.__name="Qudit_"+str(Qudit.apriori_count)

        NontangledQudit.namespace.add(self.__name)
        
        self.__N=N
        entropy=0
        super().__init__(f"NontangledQudit{N}",True,True,True,True,True,{None},entropy=entropy)
        
    @property
    def num_levels(self): return self.__N

class Float(dtype):
    count=0
    apriori_count=0
    namespace=set()
    def __init__(self,value=None,standard=32,name=None,**kwargs):
        
        
        if not (value==None or isinstance(value, (float, int, np.floating, np.integer))):
            raise ValueError("value should be a float or int")
        
        
        if not standard in {None,16,32,64}:
            raise ValueError("standard number of bits are 16, 32, 64 or None. Pass None then specify exponents, mantissa, bias arguments to implement custom precission")
        
        if standard==None:
            if not("exponent" in kwargs.keys() and "mantissa" in kwargs.keys()):
                raise ValueError("You should define number of bits of exponent and mantissa")
        
        if standard == 64:
            self.__exponent = 11
            self.__mantissa = 52
        elif standard == 32:
            self.__exponent = 8
            self.__mantissa = 23
        elif standard == 16:
            self.__exponent = 5
            self.__mantissa = 10
            
        elif standard==None:
            self.__exponent=kwargs["exponent"]
            self.__mantissa=kwargs["mantissa"]
        
        self.__depth=1+self.__exponent+self.__mantissa

        fi=self.float_info
        if value==None:
            self.__value=None
        
        elif not fi["max"]>value>fi["min"]:
            raise ValueError(f"Float should lie within range ({fi['min']}, {fi['max']})")
        
        else:
            self.__value=value

        if name in Float.namespace:raise ValueError(f"name={name} of the Float is used previously. Pick another name")
        if not (name==None or type(name)==str):raise TypeError("name of the Float is either a string or None")
        if name:
            self.__name=name
            Float.count+=1
            
        else:
            Float.count+=1
            Float.apriori_count+=1
            self.__name="Float"+str(standard)+"_"+str(Float.apriori_count)
            
        Float.namespace.add(self.__name)
        
        super().__init__("float",True,True,True,True,True,{None},entropy=self.__depth)
    


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
                    "depth": self.__depth,
                    "exp_dig": self.__exponent
                }

    def __repr__(self):
        return f"Float{self.__depth} {self.__value}, {self.__name}"
    
    @property
    def value(self):
        
        return self.__value
    
    
    def set_value(self,val):
        self.__value=val
        
    @property
    def name(self):
        return self.__name
    
    def check_value_is_appropriate(self, value):
        if not (value is None or isinstance(value, (float, int, np.floating, np.integer))):
            raise ValueError("value should be a float or int")
        return True


    

class Complex(dtype):
    count = 0
    apriori_count = 0
    namespace = set()

    def __init__(self, value=None, containers=(64, 64), name=None):
        
        if value is None:
            self.__value = None
        else:
            if isinstance(value, (list, tuple)):
                if len(value) != 2:
                    raise ValueError("If providing a list or tuple, it should contain exactly two real numbers.")
                re, im = value
            elif isinstance(value, complex):
                re, im = value.real, value.imag
            else:
                raise TypeError("Value must be either a tuple/list of real numbers or a complex number.")
            
            # Check types for re and im
            if not isinstance(re, (float, int, np.floating, np.integer)):
                raise TypeError("Real part of complex number must be a float or int")
            if not isinstance(im, (float, int, np.floating, np.integer)):
                raise TypeError("Imaginary part of complex number must be a float or int")

            self.__value = re, im


        if not isinstance(containers, (tuple, list)) or len(containers) != 2:
            raise TypeError("containers should be packed in a 2-tuple or list")
            
        if not all(isinstance(element, (int, float)) or element in {16, 32, 64} for element in containers):
            raise ValueError("Containers can be either Float class defined in DigitalSoul or specify bit numbers")
        
        
        self.__containers = tuple([t if isinstance(t, Float) else Float(None, t) for t in containers])
        

        if name in Complex.namespace:
            raise ValueError(f"name={name} of the Complex is used previously. Pick another name")
        if not (name is None or isinstance(name, str)):
            raise TypeError("name of the Complex is either a string or None")

        if name:
            self.__name = name
            Complex.count += 1
        else:
            Complex.count += 1
            Complex.apriori_count += 1
            self.__name = "Complex_" + str(Complex.apriori_count)
    
        Complex.namespace.add(self.__name)

        super().__init__("Complex", True, True, True, True, True, {None}, self.__containers[0].entropy + self.__containers[1].entropy)

    @property    
    def re(self):
        return self.__value[0] if self.__value is not None else None

    @property
    def im(self):
        return self.__value[1] if self.__value is not None else None

    @property    
    def value(self):
        return self.__value
    
    def set_value(self, value):
        if value is None:
            self.__value = None
            return
        
        if isinstance(value, (list, tuple)):
            self.__value = tuple(value)  # Convert list to tuple, if needed
        elif isinstance(value, complex):
            self.__value = value.real, value.imag

    
    @property
    def containers(self):
        return self.__containers

    @property
    def name(self):
        return self.__name
    
    def __repr__(self):
        return f"Complex {self.__value[0]} + {self.__value[1]}j, {self.__name}"
    
    def check_value_is_appropriate(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("If providing a list or tuple, it should contain exactly two real numbers.")
            re, im = value
        elif isinstance(value, complex):
            re, im = value.real, value.imag
        else:
            raise ValueError("Value must be either a tuple/list of real numbers or a complex number.")
            
        if not (isinstance(re, (float, int, np.floating, np.integer)) and isinstance(im, (float, int, np.floating, np.integer))):
            raise ValueError("Real and imaginary parts should be of type float or int.")
        return True


class ArrayHolder(dtype):
    def __init__(self, values, dtype,config={}):
        if dtype in type(dtype).__bases__:
            raise TypeError("Make sure that dtype is defined type in DigitalSoul. Known types are here:\nBool, BoolArray, Int, Float, Qubit, Qudit, Complex")
        
        self.__config=config
        config["value"]=None
        
        try:
            leading=dtype(**config)
        except Exception as e:
            raise Exception("dtype couldn't be initialized. Make sure that config is correctly loaded:"+str(e))
        
        try:
            self.__values=np.array(values)
        except Exception as e:
            raise ValueError("values should be an arraylike:"+str(e))
            
            
        
        for i in np.nditer(self.__values):
            if not leading.check_value_is_appropriate(i.item()):
                raise ValueError("All values must be appropriate for configured dtype")
            
        
        self.__configed=leading
        self.__dtype=dtype
        
        
        super().__init__(name=f"Sculk-{leading.dtype}",cpu=leading.cpu_compat,gpu= leading.gpu_compat,qubit=leading.qubit_compat,vhdl=leading.vhdl_compat,matlab=leading.matlab_compat,convertables=leading.convertables,entropy=leading.entropy*self.size)

    @property
    def size(self):
        return self.__values.size

    @property
    def values(self):
        return self.__values

    @property
    def metadata(self):
        return self.__dtype, self.__config.copy()
    
    def __getitem__(self, index):
        if not ((isinstance(index, tuple) and all(isinstance(t,slice) for t in index)) or isinstance(index,(slice,int))):
            raise TypeError("Specify a slice( obj[start:stop:step,...] to access elements")
        return (self.__values[index],self.metadata)
    
    def __setitem__(self, index, values):
        if not ((isinstance(index, tuple) and all(isinstance(t,slice) for t in index)) or isinstance(index,(int,slice))):
            raise TypeError("Specify a slice( obj[start:stop:step,...] to access elements")
            
        self.__values[index]=values

    def __repr__(self):
        return f"{self.dtype} size:{self.size}"
        


    


class Edge(object):
    edge_count=0
    apriori_edge_count=0
    namespace=set()
    def __init__(self,sculk=None,name=None):
        if name in Edge.namespace:raise ValueError(f"name={name} of the Edge is used previously. Pick another name")
        if not (name==None or type(name)==str):raise TypeError("name of the Edge is either a string or None")
        if name:
            self.__name=name
            Edge.edge_count+=1
            
        else:
            Edge.edge_count+=1
            Edge.apriori_edge_count+=1
            self.__name="Edge"+str(Edge.apriori_edge_count)
            
        Edge.namespace.add(self.__name)
        
        if dtype in type(sculk).__bases__:
            self.sculk=sculk
        else: raise TypeError("Edge can hold defined dtypes in DigitalSoul")

        
    def __repr__(self):
        return f"Edge({self.__name}) holding= {self.sculk.dtype}"
    
    def payload(self,predecessor):
        self.__predecessor=predecessor

    def unpack(self,executor="cpu"):
        if self.sculk.value==None:
            return self.__predecessor(executor)
        else:
            return self.sculk.value

        
class Node(object):
    count=0
    apriori_count=0
    namespace=set()
    def __init__(self,in_terminals,out_terminals,label=None,name=None,ops={None:None},complexity={"min":None,"max":None,"avg":None}):
        
        
        if isinstance(ops,dict):self.__ops=ops
        else: raise TypeError("ops should be a dictionary of which operation is explained in which platform")
        
        self.cpu_compat=True if "cpu" in ops.keys() else False
        self.gpu_compat=True if "gpu" in ops.keys() else False
        self.qc_compat=True if "qc" in ops.keys() else False
        self.vhdl_compat=True if "vhdl" in ops.keys() else False
        self.matlab_compat=True if "matlab" in ops.keys() else False
        self.complexity=complexity#min, max, avg
        
        if (Edge in type(in_terminals).mro()): self.__in=in_terminals,
        elif (isinstance(in_terminals, tuple) and all(Edge in type(inn).mro() for inn in in_terminals )): self.__in=in_terminals
        else: raise TypeError("in_terminals should be inherited from Edge")
        
        if (Edge in type(out_terminals).mro()):self.__out=out_terminals,
        elif (isinstance(out_terminals, tuple) and all(Edge in type(inn).mro() for inn in out_terminals )):self.__out=out_terminals
        else: raise TypeError("out_terminals should be inherited from Edge")
        
        for terminal in self.__out:
            terminal.payload(self)
        
        if name in Node.namespace:raise ValueError(f"name={name} of the Node is used previously. Pick another name")
        if not (name==None or type(name)==str):raise TypeError("name of the Node is either a string or None")
        if name:
            self.__name=name
            Node.count+=1
 
        else:
            Node.count+=1
            Node.apriori_count+=1
            self.__name="Node"+str(Node.apriori_count)
 
        Node.namespace.add(self.__name)

        self.label=label

    @property
    def ops(self):
        return self.__ops

    @property
    def name(self):
        return self.__name

    def __bind_graph(self):
        return tuple([{terminal:(self.__in, self.__ops)} for terminal in self.__out])

    def __repr__(self):
        return f"Node: op= {self.label}\ninputs= {self.__in}\nouts= {self.__out}"

    def __call__(self,executor="cpu"):
        result=self.__ops[executor](*[terminal.unpack(executor) for terminal in self.__in])
        for terminal in self.__out:
            terminal.sculk.set_value(result)
        return result


class ScalarAdd(Node):
    count=0
    def __init__(self, in_terminals,out_terminals):
        super().__init__(in_terminals,
                         out_terminals,
                         label="Add"+str(ScalarAdd.count),
                         name="Add"+str(ScalarAdd.count),
                         ops={"cpu":lambda a,b:np.add(a,b), "gpu":lambda a,b:cp.add(a,b), "tf":lambda a,b:tf.add(a,b) },
                         complexity={"min":None,"max":None,"avg":None})

        ScalarAdd.count+=1

        
class ScalarSubtract(Node):
    count=0
    def __init__(self, in_terminals,out_terminals):
        super().__init__(in_terminals,
                         out_terminals,
                         label="ScalarSubtract"+str(ScalarSubtract.count),
                         name="ScalarSubtract"+str(ScalarSubtract.count),
                         ops={"cpu": lambda a,b:np.subtract(a,b), "gpu": lambda a,b: cp.subtract(a,b),"tf": lambda a,b:tf.subtract(a,b) },
                         complexity={"min":None,"max":None,"avg":None})

        ScalarSubtract.count+=1
        

class ScalarMultiply(Node):
    count=0
    def __init__(self, in_terminals,out_terminals):
        super().__init__(in_terminals,
                         out_terminals,
                         label="ScalarMultiply"+str(ScalarMultiply.count),
                         name="ScalarMultiply"+str(ScalarMultiply.count),
                         ops={"cpu": lambda a,b:np.multiply(a,b), "gpu": lambda a,b: cp.multiply(a,b) ,"tf": lambda a,b:tf.multiply(a,b) },
                         complexity={"min":None,"max":None,"avg":None})

        ScalarMultiply.count+=1
        
class ScalarDivide(Node):
    count=0
    def __init__(self, in_terminals,out_terminals):
        super().__init__(in_terminals,
                         out_terminals,
                         label="ScalarDivide"+str(ScalarMultiply.count),
                         name="ScalarDivide"+str(ScalarMultiply.count),
                         ops={"cpu": lambda a,b:np.divide(a,b), "gpu": lambda a,b: cp.divide(a,b) ,"tf": lambda a,b:tf.divide(a,b) },
                         complexity={"min":None,"max":None,"avg":None})

        ScalarDivide.count+=1




"""
a1=Int(2)
b1=Int(7)
c1=Int(None)
e1=Int(None) 
f1=Int(None)        
a=Edge(a1)
b=Edge(b1)
c=Edge(c1)
d=Edge(Int(8))
e=Edge(e1)
f=Edge(f1)
n1=Node((a,b),c,"add",ops={"cpu":lambda a,b:np.add(a,b), "gpu":lambda a,b: cp.add(a,b), "vhdl":lambda a,b:f"{a} + {b}", "qc":"a,b CCX(a,b)"})
n2=Node((d,a),e,"subtract", ops={"cpu": lambda a,b:np.subtract(a,b), "gpu": lambda a,b: cp.subtract(a,b), "vhdl":lambda a,b:f"{a} + {b}"})
n3=Node((a,b,d),f,"fma",ops={"cpu": lambda a,b,c:np.add(a,np.multiply(b,c))})
n4=ScalarDivide((a,b), c)
print(n1("cpu"),n2("gpu"), n4("tf"))#.numpy())"""

"""
import DigitalSoul

gate0=DigitalSoul.Qudit("3", 4)
gate1=DigitalSoul.Qudit("2",4)
gate2=DigitalSoul.Qudit("0",4)
#print(gate0,"\n",gate1,"\n",gate2)

scheme={(0,1,2,3):gate0, (4,5,6,7):gate1, (8,9,10,11):gate2}
scheme_linear={0:gate0, 4:gate1, 8:gate2}


print(scheme)
print(scheme_linear)
for i in scheme_linear:
    print(i,scheme_linear[i])

ug0=DigitalSoul.NontangledQudit(scheme)
print(ug0.num_levels, gate0.num_levels)
"""
