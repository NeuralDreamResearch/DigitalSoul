#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include<cstdlib>
#include<stdexcept>
#include<string>
#include<iostream>
#include <cmath>
#include <cstdint>
#include<vector>
#include<sstream>


// Forward declaration of the QN namespace


// Forward declaration of the Complex class template within the QN namespace
namespace QN {
    template <typename T> class Complex;
}

// Forward declaration of the Qudit class template within the QN namespace
namespace QN {
    template <typename T> class Qudit;
}

// Forward declaration of the Gate class template within the QN namespace
namespace QN {
    template <typename T> class Gate;
}

// Forward declaration of the metadata namespace within the QN namespace
namespace QN {
    namespace metadata{};
}


class LUTx_1 {
private:
    size_t num_inputs;
    size_t imm; 
    size_t LogicID;
    bool residency[4]={true, false, false, false};//0 CPU-C++, 1 NVGPU-CUDA+C++, 2 FPGA-VHDL, 3 QNetwork-C++
    size_t *d_LogicID, *d_num_inputs;

public:
    LUTx_1(size_t inputs, size_t logicID)
        : num_inputs(inputs), LogicID(logicID)
        {
        	imm=num_inputs-1;

        }

    // Getter and setter methods
    size_t getNumInputs() const { return num_inputs; }
    void setNumInputs(size_t inputs) { num_inputs = inputs; }

    size_t getLogicID() const { return LogicID; }
    void setLogicID(size_t logicID) 
	{ 
		LogicID = logicID;
	}


    //Compute interface

    bool computeCPU(std::vector<bool> inputs)
    {
     
        size_t recall_index(0);
        for(size_t i(0); i<num_inputs; i++)
        {
          recall_index |= inputs[i]<<i;
        
        };
        return (LogicID>>recall_index & 1);
    }


    template <typename T=float> QN::Gate<T> UnitaryGen()
    {
    	
    	QN::Complex<T> *ops=new QN::Complex<T>[(2<<this->num_inputs)*(2<<this->num_inputs)];
		size_t maxi=2<<this->num_inputs;
		for(size_t i=0;i<maxi;i++)
		{
            size_t a0=i*maxi;
			for(size_t j=0;j<maxi;j++)
			{
                size_t a1=a0+j;
                ops[a1].re=0;
                ops[a1].im=0;
                
			}
            
		}
	
		size_t maxit=(2<<this->num_inputs)>>1;
		for(size_t i=0;i<maxit;i++)
		{
			//std::cout<<(LogicID>>i & 1)<<'\n';
	        if((LogicID>>i & 1))
	        {
		        ops[((i<<1 )+ 1)*maxi+(i<<1)].re=1.;
		        ops[(i<<1 )*maxi+(i<<1)+1].re=1.;
	        }

	        else
	        {
		        ops[(i<<1)*maxi+(i<<1)].re=1.;
		        ops[((i<<1 )+ 1)*maxi+(i<<1) + 1].re=1.;
	        }	


		}

			return QN::Gate<T>(2<<this->num_inputs,ops,1e8,true);
    };





	std::string entityGen(std::string ename="LUTx_1")
	{
	    std::string a="library IEEE;\nuse IEEE.std_logic_1164.all;\nentity "+ename+" is\nport(\ninputs: in std_logic_vector(0 to "+std::to_string(this->num_inputs-1)+");\no: out\nstd_logic);\nend "+ename+";\narchitecture structural of "+ename+" is\nbegin\nprocess(inputs)\nbegin\ncase inputs is\n";
	    for(size_t i=0;i<(1<<this->num_inputs);i++)
	    {
	        std::string a0="";
	        for(size_t j=0;j<this->num_inputs;j++)
	        {
	            a0+=std::to_string((i>>j)&1);
	        
	        }
	        a+="when \""+a0+"\" => o <= '"+std::to_string(this->LogicID>>i & 1)+"' ;\n";
	    }
	    a+="when others => o <= '0' ;\nend case;\nend process;\nend structural;";
	    
	    return a;
	
	}

    std::string ThermoTable()
    {
            std::string out="LUT";
	    out+=std::to_string(this->num_inputs)+" ID="+std::to_string(this->LogicID)+"\n";
	    for (size_t i(0);i<this->num_inputs;i++)
	    {
		    out+="______";
	    
	    }
	    out+="\n";
	    for (size_t i(0);i<num_inputs;i++)
	    {
		    out+="|in"+std::to_string(i);
	    
	    }
	    out+="|out|energy expenditure|\n";
        

	    for (size_t i(0);i<(1<<this->num_inputs);i++)
	    {
            uint64_t net=0;
		    for (size_t j(this->num_inputs-1);j!=size_t(-1);j--)
		    {
			    out+="| "+std::to_string(1 & (i>>(imm-j)))+" ";
                net+=1 & (i>>(imm-j));
			    
		    }
		    out+="| "+std::to_string(1 & (this->getLogicID()>>i))+"  | " + std::to_string((int64_t)((1 & (this->getLogicID()>>i))-net))+"|\n";
	    }
	    
	    return out;
    }
    
    
    
    // human interface
    
    std::string ss()
    {
    
      return "LookUp Table|ID="+std::to_string(LogicID)+" with "+std::to_string(num_inputs)+" inputs";
    }
    
    std::string LookUpTable()
	{
        std::string out="LUT";
	out+=std::to_string(this->num_inputs)+" ID="+std::to_string(this->LogicID)+"\n";
	for (size_t i(0);i<this->num_inputs;i++)
	{
		out+="______";
	
	}
	out+="\n";
	for (size_t i(0);i<num_inputs;i++)
	{
		out+="|in"+std::to_string(i);
	
	}
	out+="|out|\n";

	for (size_t i(0);i<(1<<this->num_inputs);i++)
	{
		for (size_t j(this->num_inputs-1);j!=size_t(-1);j--)
		{
			out+="| "+std::to_string(1 & (i>>(imm-j)))+" ";
			
		}
		out+="| "+std::to_string(1 & (this->getLogicID()>>i))+" |\n";
	}
	
	return out;
	
	}



};


namespace QN
{
	template<typename T=float> class Complex
	{
	public:
		T re, im;
		Complex(void){}; // Default constructor
		Complex(T re, T im){this->re=re; this->im=im;};
		
		void operator() (T re, T im){this->re=re; this->im=im;};
		
		Complex<T> operator +(const Complex &B)
		{
			return Complex<T>(this->re+B.re, this->im+B.im);
		};
		void operator +=(const Complex &B)
		{
			this->re+=B.re; this->im+=B.im;
		};

		Complex<T> operator -(const Complex &B)
		{
			return Complex<T>(this->re-B.re, this->im-B.im);
		};
		void operator -=(const Complex &B)
		{
			this->re-=B.re; this->im-=B.im;
		}
		
		Complex<T> operator*(const Complex &B)
		{
			return Complex<T>(this->re*B.re-this->im*B.im, this->re*B.im+this->im*B.re);
		};
		void operator*=(const Complex &B)
		{
			 this->re=this->re*B.re-this->im*B.im;
			 this->im=this->re*B.im+this->im*B.re;
		};
		
		Complex<T> operator/(const Complex &B)
		{
			T magB=B.re*B.re + B.im*B.im;
			if(magB!=0){return Complex<T>((this->re*B.re + this->im * B.im)/magB, (this->im * B.re - this->re * B.im)/magB);}
			else{throw std::runtime_error("ZeroDivisionError: Invaild value encountered in Complex division");}
		};
		void operator/=(const Complex &B)
		{
			T magB=B.re*B.re + B.im*B.im;
			if(magB!=0){this->re=(this->re*B.re + this->im * B.im)/magB; this->im=(this->im * B.re - this->re * B.im)/magB;}
			else{throw std::runtime_error("ZeroDivisionError: Invaild value encountered in Complex division");}
		};
		
		T magnitude(void)
		{
			
			return sqrt(this->re*this->re+this->im*this->im);
		};
		
		T arg(void)
		{
			return atan2(this->im,this->re);
		};
		
		Complex<T> conj()
		{
			return Complex<T>(this->re, -1*(this->im));
		}
		
		std::string ss(void)
		{
			if(this->im>=0)
			{
				return std::to_string(this->re)+"+"+std::to_string(this->im)+"j";
			}
			else
			{
				return std::to_string(this->re)+"-"+std::to_string(-1*this->im)+"j";
			};
		};
	};



	template <typename T=float> class Qudit
	{

		size_t N_system=2;
		Complex<T> *statevector=nullptr;
	public:
		Qudit(){};
		Qudit(size_t N_system, bool allocate_now=true)
		{
			this->N_system=N_system;
			if(allocate_now){this->statevector=new Complex<T>[this->N_system];this->statevector[0](1,0);};
			
		};
		
		Qudit(size_t N_system, Complex<T> *new_statevector, T epsilon=1e-8, bool trust=false)
		{
			if(trust){delete [] this->statevector;this->statevector=new_statevector;}
			else
				{
				T cum_sum=0;
				for(size_t i=0;i<this->N_system;i++)
				{
					cum_sum+=new_statevector[i].magnitude();
				}
			
				if(abs(cum_sum-1)>epsilon)
				{
					throw std::runtime_error("DenormalizedPureState: Invalid value encountered in new_statevector. The new_statevector must be normalized to one. Consider to normalize or modify components");
				}
			
				delete [] this->statevector;
				this->statevector=new_statevector;
				}
		};
		
		void oneHot(size_t N)
		{
			for(size_t i=0;i<this->N_system;i++)
			{
				this->statevector[i](0,0);
			}
			this->statevector[N](1,0);
		};
		
		void loadStatevector(Complex<T> *new_statevector, T epsilon=1e-8, bool trust=false)
		{
			if(trust){delete [] this->statevector;this->statevector=new_statevector;}
			else
				{
				T cum_sum=0;
				for(size_t i=0;i<this->N_system;i++)
				{
					cum_sum+=new_statevector[i].magnitude();
				}
			
				if(abs(cum_sum-1)>epsilon)
				{
					throw std::runtime_error("DenormalizedPureState: Invalid value encountered in new_statevector. The new_statevector must be normalized to one. Consider to normalize or modify components");
				}
			
				delete [] this->statevector;
				this->statevector=new_statevector;
				}
		};
		
		void freeStatevector(void)
		{	
			delete [] (this->statevector);
			this->statevector=nullptr;
		};
		
		Complex<T> *Psi(void)
		{return (this->statevector);};
		
		size_t numStates(void)
		{
			return this->N_system;
		};
		
		std::string ss(bool verbosity=false)
		{
			if(verbosity)
			{
				std::string rs=std::to_string(this->N_system)+"-level Qudit\n\u03A8 = (";
				for(size_t i=0;i<this->N_system-1;i++)
				{
					rs+=this->statevector[i].ss()+", ";
				}
				rs+=this->statevector[this->N_system-1].ss()+") ";
				return rs;
			}
			
			else{return std::to_string(this->N_system)+"-level Qudit";};
		};
	};

	template<typename T> bool isUnitary(Complex<T> *A, size_t &operandDim, T epsilon=1e-8)
	{
		
		for(size_t i=0;i<operandDim;i++)
		{
			for(size_t j=0;j<operandDim; j++)
			{
				Complex<T> current(0,0);
				for(size_t k=0;k<operandDim;k++)
				{
					current+=A[i*operandDim+k]*(A[j*operandDim+k].conj());
				}//j,k -> kj
				if(i==j){if(abs(current.re-1)<epsilon && abs(current.im)<epsilon){} else{return false;}}
				else{if(abs(current.re)<epsilon && abs(current.im)<epsilon){}else{return false;}}		
			}
		}
		
		return true;
	}

	template<typename T=float> void sqMatmul(Complex<T> *U1, Complex<T> *U2, Complex<T> *U3, size_t operandDim)
	{
		for(size_t i=0; i<operandDim;i++)
		{
			size_t ai=i*operandDim;
			for(size_t j=0;j<operandDim;j++)
			{
				for(size_t k=0;k<operandDim;k++)
				{
					U3[ai+j]+=U1[ai+k]*U2[k*operandDim+j];
				}
				
			}
		}
		
	}
	template<typename T=float> void sqMVDot(Complex<T> *U, Complex<T> *S1, Complex<T> *S2, size_t operandDim)
	{
		for(size_t i=0;i<operandDim;i++)
		{
			size_t ai=operandDim*i;
			for(size_t j=0;j<operandDim;j++)
			{
				S2[i]+=U[ai+j]*S1[j];
			}
			
		}
	}

	template <typename T=float> class Gate
	{
	private:
		Complex<T> *data=nullptr;
		size_t operandDim=2;
	public:
		Gate(){};
		Gate(size_t operandDim){this->operandDim=operandDim;};
		Gate(size_t OperandDim, Complex<T> *Operator, T epsilon=1e-8, bool trust=false)
		{
			this->operandDim=OperandDim;
			if(trust)
			{
				this->data=Operator;
			}
			else
			{
				if(isUnitary(Operator,this->operandDim, epsilon)){this->data=Operator;}		
				else{throw std::runtime_error("Operator is not Unitary. Try to change either coefficients or accomodate a change for epsilon so that it may not be detected precisely");}
				
			}
		};

		void loadOperator(Complex<T> *Operator, T epsilon=1e-8, bool trust=false)
		{
			if(trust)
			{
				this->data=Operator;
			}
			else
			{
				if(isUnitary(Operator,this->operandDim, epsilon)){this->data=Operator;}		
				else{throw std::runtime_error("Operator is not Unitary. Try to change either coefficients or epsilon so that it may not be detected precisely");}
				
			}
		};
		void transform(Qudit<T> &sv, bool trust=false)// CONTINUE	FLAG
		{
			if(!trust)
			{
				if(sv.numStates()!=this->operandDim)
				{
					throw std::runtime_error("Unmatched dimensions of operator and operand. Gate is "+std::to_string(this->operandDim)+"-level and Qudit is "+std::to_string(sv.numStates())+"-level system");
				}
			}
			
			Complex<T> *s1=new Complex<T>[this->operandDim];
			Complex<T> *s2=sv.Psi();
			for(size_t i=0;i<this->operandDim;i++)
			{
				s1[i]=s2[i];
			}
			
			for(size_t i=0;i<this->operandDim;i++)
			{
				s2[i].re=0.;
				s2[i].im=0.;
			}
		
			sqMVDot(this->data, s1, s2, this->operandDim);
		}
		
		std::string ss()
		{
		  std::string out="Quantum Gate on "+std::to_string(this->operandDim) +"-level system \nU=[";
		  for(size_t i=0;i<this->operandDim;i++)
		  {
		    out+='[';
		    for(size_t j=0;j<this->operandDim-1;j++)
		      {
			out+=this->data[i*this->operandDim+j].ss()+',';
		      
		      }
		      out+=this->data[(i+1)*this->operandDim-1].ss()+']'+','+'\n';
		  
		  }
		  out+=']';
		return out;
		};
		
	};
	
	namespace metadata
	{
		typedef float native;
		Complex<native> Xarray[4]={Complex<native>(0,0),Complex<native>(1,0), Complex<native>(1,0), Complex<native>(0,0) } ;

		Complex<native> Yarray[4]={Complex<native>(0,0),Complex<native>(0,-1), Complex<native>(0,1), Complex<native>(0,0) } ;

		Complex<native> Zarray[4]={Complex<native>(1,0),Complex<native>(0,0), Complex<native>(0,0), Complex<native>(-1,0) } ;
	}

};


struct residency
{
	private:
		bool* res; // Pointer to dynamically allocated array representing all resources (CPU, GPU, Signal, Register, BRAM, QNetwork)

	public:
		// Constructor with default arguments
		residency(bool cpu = true, bool gpu = false, bool signal = false, bool registerRes = false, bool bram = false, bool QN = false) {
		    res = new bool[6]; // Allocate memory for the unified resources array
		    res[0] = cpu;      // Initialize the elements with the provided values
		    res[1] = gpu;
		    res[2] = signal;
		    res[3] = registerRes;
		    res[4] = bram;
		    res[5] = QN;
		}

		// Overloaded constructor to initialize with a boolean array
		residency(const std::array<bool, 6>& initArray) {
		    res = new bool[6]; // Allocate memory for the unified resources array
		    for (int i = 0; i < 6; ++i) {
		        res[i] = initArray[i]; // Initialize with array values
		    }
		}

		// Destructor to clean up the dynamically allocated memory
		~residency() {
		    delete[] res; // Free the unified resources array
		}

		// Query Methods

		bool isCPUAvailable() const {
		    return res[0];
		}

		bool isGPUAvailable() const {
		    return res[1];
		}

		bool isSignalAvailable() const {
		    return res[2];
		}

		bool isRegisterAvailable() const {
		    return res[3];
		}

		bool isBRAMAvailable() const {
		    return res[4];
		}

		bool isQNetworkAvailable() const {
		    return res[5];
		}

		// Setter Methods

		void setCPUAvailability(bool available) {
		    res[0] = available;
		}

		void setGPUAvailability(bool available) {
		    res[1] = available;
		}

		void setSignalAvailability(bool available) {
		    res[2] = available;
		}

		void setRegisterAvailability(bool available) {
		    res[3] = available;
		}

		void setBRAMAvailability(bool available) {
		    res[4] = available;
		}

		void setQNetworkAvailability(bool available) {
		    res[5] = available;
		}

		// Method to export the state as a boolean array
		std::array<bool, 6> toBoolArray() const {
		    return { res[0], res[1], res[2], res[3], res[4], res[5] };
		}

		// Function to display the current state (for demonstration purposes)
		void ss() const {
		    std::cout << "CPU: " << (res[0] ? "Available" : "Not Available") << ", ";
		    std::cout << "GPU: " << (res[1] ? "Available" : "Not Available") << ", ";
		    std::cout << "Signal: " << (res[2] ? "Available" : "Not Available") << ", ";
		    std::cout << "Register: " << (res[3] ? "Available" : "Not Available") << ", ";
		    std::cout << "BRAM: " << (res[4] ? "Available" : "Not Available") << ", ";
		    std::cout << "QNetwork: " << (res[5] ? "Available" : "Not Available") << std::endl;
		}
};


struct dtype {
public:
    uint8_t type_id;
    std::vector<uint8_t> descriptor;

    // Constructor for initializing type_id and descriptor
    dtype(uint8_t type_id, const std::vector<uint8_t>& descriptor = {}) : type_id(type_id), descriptor(descriptor) {
        // Set the default descriptor based on type_id if not provided
        switch (type_id) {
            case 0: // bool
                this->descriptor = {};
                break;
            case 1: // uint
            case 2: // signed integer
                if (descriptor.empty()) {
                    this->descriptor = {32}; // Default to 32 bits if not specified
                }
                break;
            case 3: // float
                if (descriptor.empty()) {
                    // Default to 32-bit float with 8 bits for exponent and 23 bits for mantissa
                    this->descriptor = {8, 23};
                }
                break;
            case 4: // complex
                if ((descriptor.size() != 2 && descriptor[0] != 3) || (descriptor.size() != 3 && descriptor[0] == 3)) {
                    std::cerr << "Error: Invalid descriptor for complex type. "
                              << "Expected (type ID, bits) for integer types or (type ID, exp bits, mantissa bits) for float type.\n";
                }
                break;
            default:
                std::cerr << "Error: Invalid type_id provided.\n";
                this->descriptor = {};
                break;
        }
    }

    // Function to print type information
    void ss() const {
        switch (type_id) {
            case 0:
                std::cout << "Type: bool\n";
                break;
            case 1:
                std::cout << "Type: uint, Num bits: " << static_cast<int>(descriptor[0]) << "\n";
                break;
            case 2:
                std::cout << "Type: signed int, Num bits: " << static_cast<int>(descriptor[0]) << "\n";
                break;
            case 3:
                std::cout << "Type: float, Exponent bits: " << static_cast<int>(descriptor[0])
                          << ", Mantissa bits: " << static_cast<int>(descriptor[1]) << "\n";
                break;
            case 4:
                std::cout << "Type: complex\n";
                if (descriptor.size() == 2) {
                    std::cout << "Subpart type ID: " << static_cast<int>(descriptor[0]) << "\n";
                    std::cout << "Subpart bits: " << static_cast<int>(descriptor[1]) << "\n";
                } else if (descriptor.size() == 3) {
                    std::cout << "Subpart type ID: " << static_cast<int>(descriptor[0]) << "\n";
                    std::cout << "Subpart exponent bits: " << static_cast<int>(descriptor[1]) << "\n";
                    std::cout << "Subpart mantissa bits: " << static_cast<int>(descriptor[2]) << "\n";
                } else {
                    std::cout << "Invalid descriptor provided for complex type.\n";
                }
                break;
            default:
                std::cout << "Unknown type\n";
                break;
        }
    }
    
    // Method to get the number of bits
    int get_num_bits() const {
        switch (type_id) {
            case 0:
                return 1; // Boolean is typically 1 bit, but it can vary based on context
            case 1:
            case 2:
                return static_cast<int>(descriptor[0]);
            case 3:
                return static_cast<int>(descriptor[0]) + static_cast<int>(descriptor[1]) + 1; // Including the sign bit
            case 4:
                if (descriptor.size() == 2) {
                    // For complex types with integer or unsigned integer subparts
                    return static_cast<int>(descriptor[1]) * 2; // Real + Imaginary parts
                } else if (descriptor.size() == 3) {
                    // For complex types with float subparts
                    return (static_cast<int>(descriptor[1]) + static_cast<int>(descriptor[2]) + 1) * 2;
                }
                return 0; // Unknown or insufficient descriptor
            default:
                return 0; // Unknown type
        }
    }
    
    // Method to get the display name
    std::string get_display_name() const {
        std::ostringstream oss;
        switch (type_id) {
            case 0:
                oss << "Bool";
                break;
            case 1:
                oss << "UInt" << static_cast<int>(descriptor[0]);
                break;
            case 2:
                oss << "Int" << static_cast<int>(descriptor[0]);
                break;
            case 3:
                // Float{num_bits}E{num_exp}M{num_mantissa} standard
                oss << "Float" << get_num_bits() << "E" << static_cast<int>(descriptor[0]) << "M" << static_cast<int>(descriptor[1]);
                break;
            case 4:
                oss << "Complex of ";
                if (descriptor.size() == 2) {
                    uint8_t sub_type_id = descriptor[0];
                    switch (sub_type_id) {
                        case 1:
                            oss << "UInt" << static_cast<int>(descriptor[1]);
                            break;
                        case 2:
                            oss << "Int" << static_cast<int>(descriptor[1]);
                            break;
                        default:
                            oss << "Unknown";
                            break;
                    }
                } else if (descriptor.size() == 3) {
                    if (descriptor[0] == 3) {
                        // Float{num_bits}E{num_exp}M{num_mantissa} standard for subpart
                        oss << "Float"
                            << (static_cast<int>(descriptor[1]) + static_cast<int>(descriptor[2]) + 1)
                            << "E" << static_cast<int>(descriptor[1])
                            << "M" << static_cast<int>(descriptor[2]);
                    } else {
                        oss << "Unknown";
                    }
                } else {
                    oss << "Unknown";
                }
                break;
            default:
                oss << "Unknown";
                break;
        }
        return oss.str();
    }
    
    std::string vhdl_descriptor() const
    {
        switch(this->type_id)
        {
            case 0:
                return "std_logic";
                break;
            case 1:
                return "unsigned("+std::to_string(this->descriptor[0] -1)+" downto 0)";
                break;
            case 2:
                return "signed("+std::to_string(this->descriptor[0] -1)+" downto 0)";
                break;
            case 3:
                return "std_logic_vector("+std::to_string(this->get_num_bits())+" downto 0)";
                break;
        }
    
    }
};



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
        
    // Expose the residency struct
    py::class_<residency>(m, "residency")
        .def(py::init<bool, bool, bool, bool, bool, bool>(), 
             py::arg("cpu") = true, 
             py::arg("gpu") = false, 
             py::arg("signal") = false, 
             py::arg("registerRes") = false, 
             py::arg("bram") = false, 
             py::arg("QN") = false) // Constructor with default arguments
        .def(py::init<const std::array<bool, 6>&>(), py::arg("initArray")) // Overloaded constructor with boolean array

        .def("isCPUAvailable", &residency::isCPUAvailable)     // Query method for CPU availability
        .def("isGPUAvailable", &residency::isGPUAvailable)     // Query method for GPU availability
        .def("isSignalAvailable", &residency::isSignalAvailable)   // Query method for Signal availability
        .def("isRegisterAvailable", &residency::isRegisterAvailable)   // Query method for Register availability
        .def("isBRAMAvailable", &residency::isBRAMAvailable)   // Query method for BRAM availability
        .def("isQNetworkAvailable", &residency::isQNetworkAvailable) // Query method for QNetwork availability

        .def("setCPUAvailability", &residency::setCPUAvailability, py::arg("available")) // Setter for CPU availability
        .def("setGPUAvailability", &residency::setGPUAvailability, py::arg("available")) // Setter for GPU availability
        .def("setSignalAvailability", &residency::setSignalAvailability, py::arg("available")) // Setter for Signal availability
        .def("setRegisterAvailability", &residency::setRegisterAvailability, py::arg("available")) // Setter for Register availability
        .def("setBRAMAvailability", &residency::setBRAMAvailability, py::arg("available")) // Setter for BRAM availability
        .def("setQNetworkAvailability", &residency::setQNetworkAvailability, py::arg("available")) // Setter for QNetwork availability

        .def("toBoolArray", &residency::toBoolArray) // Method to export the state as a boolean array
        .def("ss", &residency::ss); // Method to display the state of resources


    // Expose the dtype class
    py::class_<dtype>(m, "dtype")
        .def(py::init<uint8_t, const std::vector<uint8_t>&>(), py::arg("type_id"), py::arg("descriptor") = std::vector<uint8_t>())
        .def_readwrite("type_id", &dtype::type_id)
        .def_readwrite("descriptor", &dtype::descriptor)
        .def("ss", &dtype::ss)
        .def("__repr__", &dtype::get_display_name)
        .def("get_num_bits", &dtype::get_num_bits)
        .def("vhdl_descriptor", &dtype::vhdl_descriptor);

}

