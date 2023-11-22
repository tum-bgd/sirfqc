import cirq
import sympy


class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


def create_fvqc(observable):
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    circuit = cirq.Circuit()

    # Prepare the readout qubit
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits=data_qubits,
        readout=readout)

    # Add layers
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Prepare measurement
    if observable == 'x':
        circuit.append(cirq.H(readout))

    if observable == 'y':
        s_dg = cirq.ops.ZPowGate(exponent=-(1 / 2))
        circuit.append(s_dg(readout))
        circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)
