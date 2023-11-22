import cirq
import sympy


class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_full_layer(self, circuit, gate1, gate2, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            ry = gate1(rads=symbol)
            circuit.append(ry(qubit))
        k = 0
        j = 2
        for i in range(len(self.data_qubits)):
            if i + k + 1 < len(self.data_qubits):
                circuit.append(gate2(self.data_qubits[i + k], self.data_qubits[i + k + 1]))
                k += 3
            if i + j + 1 < len(self.data_qubits):
                circuit.append(gate2(self.data_qubits[i + j + 1], self.data_qubits[i + j]))
                j += 3

    def add_layer(self, circuit, remains, gate1, gate2, prefix):
        qubits = self.data_qubits
        del qubits[3:len(self.data_qubits):4]
        del qubits[3:len(self.data_qubits):3]
        del qubits[0]

        for i, qubit in enumerate(qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            ry = gate1(rads=symbol)
            circuit.append(ry(qubit))
        k = 0
        j = 2
        for i in range(len(self.data_qubits)):
            if i + k + 1 < len(self.data_qubits):
                circuit.append(gate2(self.data_qubits[i + k], self.data_qubits[i + k + 1]))
                k += 3
            if i + j + 1 < len(self.data_qubits):
                circuit.append(gate2(self.data_qubits[i + j + 1], self.data_qubits[i + j]))
                j += 3


def create_gvqc(observable):
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = data_qubits[10]

    circuit = cirq.Circuit()

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits, readout=readout)

    # Add layers
    builder.add_full_layer(circuit, cirq.Ry, cirq.CNOT, "ry0")
    k = 1
    remains = len(data_qubits) / 2
    while remains > 1:
        builder.add_layer(circuit, remains, cirq.Ry, cirq.CNOT, "ry" + str(k))
        remains = remains / 2
        k += 1

    symbol = sympy.Symbol("ry" + str(k) + '-' + str(0))
    gate = cirq.Ry(rads=symbol)
    circuit.append(gate(readout))

    # Prepare measurement
    if observable == 'x':
        circuit.append(cirq.H(readout))

    if observable == 'y':
        s_dg = cirq.ops.ZPowGate(exponent=-(1 / 2))
        circuit.append(s_dg(readout))
        circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)
