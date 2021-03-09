import numpy as np
class Regresor:
    def __init__(self, regresor):
        self.regressor = regresor
        
    def calculaOptimo(self, datos_entrada):
        datos = np.array(datos_entrada)
        datos = datos.reshape(1, -1)
        datos_salida = self.regressor.predict(datos)
        return datos_salida
    
    def imprimeOptimo(self, datos_entrada):
        datos = np.array(datos_entrada)
        datos = datos.reshape(1, -1)
        EE = self.regressor.predict(datos)[0][0]
        EC = self.regressor.predict(datos)[0][1]
        cost = self.regressor.predict(datos)[0][2]
        print('EE: %s\nEC: %s\nCosto Ponderado unitario: %s'% (EE,EC,cost))