import cirq
import sympy
import math

class CircuitLayerBuilder_svqc():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout    
        
    def add_singleqt_layer(self, circuit, qubits, gate, prefix):        
        for i, qubit in enumerate(qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            ry = gate(rads=symbol)
            circuit.append(ry(qubit))
    
    
    def add_twoqt_layer(self, circuit, qubits, rangeparam, gate, prefix):   
        controls=[]
        targets=[]
        r=rangeparam
        n=len(self.data_qubits)
        for j in range(1,int((n)/math.gcd(n,r))):
            controls.append((j*r)%n)
            targets.append((j*r-r)%n)
        controls.append(0)
        targets.append(controls[n-2])

        for i in range(0, n):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubits[controls[n-1-i]], qubits[targets[n-1-i]])**symbol)

    
def create_svqc(observable, grid=[4, 4]):
    data_qubits = cirq.GridQubit.rect(int(grid[0]), int(grid[0]))
    readout = data_qubits[0]
    circuit = cirq.Circuit()

    builder = CircuitLayerBuilder_svqc(
        data_qubits=data_qubits, readout=readout)

    # Add layers
    k = 0
    builder.add_singleqt_layer(circuit, data_qubits, cirq.Rz, "rz" + str(k))
    builder.add_singleqt_layer(circuit, data_qubits, cirq.Rx, "rx" + str(k))
    builder.add_twoqt_layer(circuit, data_qubits, 1, cirq.CNOT,  "cnot" + str(k))
    k+=1
    builder.add_singleqt_layer(circuit, data_qubits, cirq.Rz, "rz" + str(k))
    builder.add_singleqt_layer(circuit, data_qubits, cirq.Rx, "rx" + str(k))
    builder.add_twoqt_layer(circuit, data_qubits, 3, cirq.CNOT,  "cnot" + str(k))

    # Prepare readout
    symbol = sympy.Symbol('rx2' + '-' + str(0))
    rz = cirq.Rz(rads=symbol)
    circuit.append(rz(readout))

    # Prepare measurement
    if observable == 'x':
        circuit.append(cirq.H(readout))

    if observable == 'y':
        s_dg = cirq.ops.ZPowGate(exponent=-(1 / 2))
        circuit.append(s_dg(readout))
        circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)