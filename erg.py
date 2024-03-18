try: import DigitalSoul.dscpp
except:print("DigitalSoul C++ assets are not compiled well")

class ThermodynamicGate(DigitalSoul.dscpp.LUTx_1):
    def __init__(self, num_inputs, logic_id,thermo_enc={0:0, 1:1}):
        super().__init__(num_inputs, logic_id)
        self.__thermo_enc=thermo_enc
    

    @property
    def thermo_enc(self):return self.__thermo_enc



    

