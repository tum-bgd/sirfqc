"""Quantum encodings"""
import numpy as np
import cirq


"""Basis-encoding"""
def basis_embedding(image):
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


"""Angle-encoding"""
def angle_embedding(image, param):
    values = np.ndarray.flatten(image)
    values = values * 2 * np.pi
    qubits = cirq.GridQubit.rect(4, 4)
    if param == 'x':
        circuit = cirq.Circuit(cirq.rx(v)(q) for v, q in zip(values, qubits))
    if param == 'y':
        circuit = cirq.Circuit(cirq.ry(v)(q) for v, q in zip(values, qubits))
    if param == 'z':
        circuit = cirq.Circuit(cirq.rz(v)(q) for v, q in zip(values, qubits))
    if param == None:
        print('Chose parameter for angle embedding! rx, ry, rz')
    return circuit
