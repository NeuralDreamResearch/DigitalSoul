#include<cstdlib>
#include<stdexcept>
#include<string>
#include<iostream>
#include <cmath>
#include <cstdint>
#include<vector>

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
