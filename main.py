import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Quantum Gates Implementation
class QuantumGate:
    @staticmethod
    def hadamard():
        return torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    
    @staticmethod
    def rx(theta):
        return torch.tensor([
            [torch.cos(theta / 2), -1j * torch.sin(theta / 2)], 
            [-1j * torch.sin(theta / 2), torch.cos(theta / 2)]
        ], dtype=torch.complex64)
    
    @staticmethod
    def cnot():
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64)

# Quantum Ecological Model with Quantum Neural Network
class QuantumEcologicalModel(nn.Module):
    def __init__(self, n_species, n_strategies):
        super().__init__()
        self.n_species = n_species
        self.n_strategies = n_strategies
        self.theta = nn.Parameter(torch.randn(n_species, n_strategies))
        
        # Classical neural network to assist in strategy adaptation
        self.classical_nn = nn.Sequential(
            nn.Linear(2 ** n_species, 128),  # Input dimension corrected
            nn.ReLU(),
            nn.Linear(128, n_species)  # Output dimension changed to n_species
        )

    def forward(self, x):
        bsz = x.shape[0]
        state = torch.zeros((2 ** self.n_species,), dtype=torch.complex64)
        state[0] = 1  # Start in |0> state
        
        # Apply Hadamard gate to each qubit
        for i in range(self.n_species):
            state = self.apply_gate(state, QuantumGate.hadamard(), i)
        
        # Apply RX gate to each qubit based on theta
        for i in range(self.n_species):
            for j in range(self.n_strategies):
                theta_ij = self.theta[i, j]
                state = self.apply_gate(state, QuantumGate.rx(theta_ij), i)
        
        # Apply entanglement (CNOT) between each pair of qubits
        for i in range(self.n_species):
            for j in range(i + 1, self.n_species):
                state = self.apply_cnot(state, i, j)
        
        # Measurement simulation (calculate probabilities)
        probs = torch.abs(state) ** 2
        probs = probs.view(1, -1)  # Flatten the tensor

        # Use classical NN to refine strategies
        refined_strategies = self.classical_nn(probs)
        return refined_strategies

    def apply_gate(self, state, gate, qubit):
        full_gate = torch.eye(2 ** self.n_species, dtype=torch.complex64)
        for i in range(2 ** self.n_species):
            for j in range(2 ** self.n_species):
                if (i >> qubit) % 2 == (j >> qubit) % 2:
                    full_gate[i, j] = gate[(i >> qubit) % 2, (j >> qubit) % 2]
        return torch.matmul(full_gate, state)
    
    def apply_cnot(self, state, control_qubit, target_qubit):
        full_cnot = torch.eye(2 ** self.n_species, dtype=torch.complex64)
        control_mask = 1 << control_qubit
        target_mask = 1 << target_qubit
        for i in range(2 ** self.n_species):
            if i & control_mask:
                j = i ^ target_mask
                full_cnot[i, i] = 0
                full_cnot[j, j] = 0
                full_cnot[i, j] = 1
                full_cnot[j, i] = 1
        return torch.matmul(full_cnot, state)

def main():
    n_species = 4  # Reduced for simplicity in tensor size
    n_strategies = 3
    model = QuantumEcologicalModel(n_species, n_strategies)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Load population data
    data_path = 'animal_populations.csv'
    if not os.path.exists(data_path):
        print(f"{data_path} not found.")
        return
    
    population_data = pd.read_csv(data_path)
    
    # Ensure the data is properly formatted
    if 'Year' not in population_data.columns or 'Species1' not in population_data.columns:
        print("CSV file must contain 'Year' and 'Species1' columns.")
        return

    # Simulation loop
    losses = []
    for epoch in range(100):
        data = torch.randn((1, n_species))  # Adjusted shape for batch size
        output = model(data)
        
        # Print shapes for debugging
        print(f"Data shape: {data.shape}")
        print(f"Output shape: {output.shape}")
        
        # Loss calculation and backward pass
        optimizer.zero_grad()
        target = torch.tensor(population_data.iloc[epoch % len(population_data)].drop('Year')).float().view(1, -1)
        print(f"Target shape: {target.shape}")  # Print target shape for debugging
        
        loss = torch.mean((output - target)**2)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Analysis and comparison of results
    plt.plot(population_data['Year'], population_data['Species1'], label='Actual Population')
    plt.plot(np.arange(100), output.detach().numpy().flatten(), label='Simulated Population')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
