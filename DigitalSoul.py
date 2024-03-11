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
    def __init__(self, value, exponent=8, mantissa=23 ):
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

class ArrayHolder(object):
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
        self.__c=ArrayHolder.count
        ArrayHolder.count+=1

    @property
    def entropy(self):
        if type(self.value)!=np.ndarray:
            self.dtype.value=None
            return self.dtype.entropy*np.prod(self.shape)
        else:return 0

    def __repr__(self):
        if type(self.value)!=np.ndarray:
            return f"ArrayHolder_{self.__c}: array of {repr(self.dtype)[:repr(self.dtype).index(' ')]} shape={self.shape} entropy={self.entropy}"
        else:
            return f"ArrayHolder_{self.__c}: array of {repr(self.dtype)[:repr(self.dtype).index(' ')]} shape={self.value.shape} entropy={self.entropy}"
    @property
    def name(self): return f"ArrayHolder_{self.__c}"
    
            
                
class Edge(object):
    count=0
    def __init__(self,sculk):
        self.sculk=sculk
        self.__c=0
        Edge.count+=1

    def vhdl(self):
        if type(self.sculk)==Bool:
            if self.sculk.value==None:
                return f"{self.sculk.name}:std_logic:='X'"
            elif self.sculk.value==0:
                return f"{self.sculk.name}:std_logic:='0'"
            elif self.sculk.value==1:
                return f"{self.sculk.name}:std_logic:='1'"
        
