import cirq
import sympy


class CircuitLayerBuilder_mera():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_ry_layer(self, circuit, prefix):
        k = 1
        for i in range(1, len(self.data_qubits), 4):
            symbol1 = sympy.Symbol(prefix + '-' + str(k))
            symbol2 = sympy.Symbol(prefix + '-' + str(k+1))
            ry1 = cirq.Ry(rads=symbol1)
            ry2 = cirq.Ry(rads=symbol2)
            circuit.append(ry1(self.data_qubits[i]))
            circuit.append(ry2(self.data_qubits[i+1]))
            k+=4
                       
    def add_cnot_layer(self, circuit):
        for i in range(0, len(self.data_qubits), 4):
            circuit.append(cirq.CNOT(self.data_qubits[i], self.data_qubits[i+1]))

        for i in range(2, len(self.data_qubits), 4):
            circuit.append(cirq.CNOT(self.data_qubits[i+1], self.data_qubits[i]))

            
    def add_cnot(self, circuit, qubit1, qubit2):
        circuit.append(cirq.CNOT(qubit1, qubit2))


    def add_ry(self, circuit, qubit, prefix, k):
        symbol = sympy.Symbol(prefix + '-' + str(k))
        ry = cirq.Ry(rads=symbol)
        circuit.append(ry(self.data_qubits[qubit]))

        
def create_mera(observable, grid=[4,4]):
    data_qubits = cirq.GridQubit.rect(int(grid[0]), int(grid[1]))
    readout = data_qubits[10]
    circuit = cirq.Circuit()

    builder = CircuitLayerBuilder_mera(
        data_qubits=data_qubits,
        readout=readout)

    # Add layers
    n=0
    builder.add_ry_layer(circuit, 'ry' + str(n))
    n+=1
    builder.add_ry(circuit=circuit, qubit=0, prefix='ry' + str(n), k=0)
    builder.add_cnot(circuit, data_qubits[1], data_qubits[2])
    builder.add_ry(circuit, 3, 'ry' + str(n), 3)
    builder.add_ry(circuit, 4, 'ry' + str(n), 4)
    builder.add_cnot(circuit, data_qubits[5], data_qubits[6])
    builder.add_ry(circuit, 7, 'ry' + str(n), 7)
    builder.add_ry(circuit, 8, 'ry' + str(n), 8)
    builder.add_cnot(circuit, data_qubits[10], data_qubits[9])
    builder.add_ry(circuit, 11, 'ry' + str(n), 11)
    builder.add_ry(circuit, 12, 'ry' + str(n), 12)
    builder.add_cnot(circuit, data_qubits[14], data_qubits[13])
    builder.add_ry_layer(circuit, 'ry' + str(n))
    builder.add_ry(circuit=circuit, qubit=15, prefix='ry' + str(n), k=15)

    builder.add_cnot_layer(circuit)
    
    n+=1
    builder.add_ry_layer(circuit, 'ry' + str(n))

    builder.add_cnot(circuit, data_qubits[2], data_qubits[5])
    builder.add_cnot(circuit, data_qubits[13], data_qubits[10])

    n+=1
    builder.add_ry(circuit, 2,' ry' + str(n), 0)
    builder.add_ry(circuit, 5,' ry' + str(n), 1)
    builder.add_ry(circuit, 10,' ry' + str(n), 2)
    builder.add_ry(circuit, 13,' ry' + str(n), 3)

    
    builder.add_cnot(circuit, data_qubits[1], data_qubits[2])
    builder.add_cnot(circuit, data_qubits[6], data_qubits[5])
    builder.add_cnot(circuit, data_qubits[9], data_qubits[10])
    builder.add_cnot(circuit, data_qubits[14], data_qubits[13])

    n+=1
    builder.add_ry(circuit, 2,' ry' + str(n), 0)
    builder.add_ry(circuit, 5,' ry' + str(n), 1)
    builder.add_ry(circuit, 10,' ry' + str(n), 2)
    builder.add_ry(circuit, 13,' ry' + str(n), 3)
    
    builder.add_cnot(circuit, data_qubits[2], data_qubits[5])
    builder.add_cnot(circuit, data_qubits[13], data_qubits[10])
    
    n+=1
    builder.add_ry(circuit, 5,' ry' + str(n), 1)
    builder.add_ry(circuit, 10,' ry' + str(n), 3)
    
    builder.add_cnot(circuit, data_qubits[5], data_qubits[10])

    builder.add_ry(circuit, 10,' ry' + str(n), 0)

    # Prepare measurement
    if observable == 'x':
        circuit.append(cirq.H(readout))

    if observable == 'y':
        s_dg = cirq.ops.ZPowGate(exponent=-(1 / 2))
        circuit.append(s_dg(readout))
        circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)