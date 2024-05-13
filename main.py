import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from sklearn.preprocessing import MinMaxScaler  # Added import for MinMaxScaler

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
            nn.Linear(2 ** n_species, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_species)
        )

    def forward(self, x):
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

def visualize_qubits(n_species):
    G = nx.DiGraph()
    nodes = range(2 ** n_species)
    G.add_nodes_from(nodes)
    
    # Add edges with weights
    for i in nodes:
        for j in nodes:
            if i != j:
                weight = np.random.randint(1, 100)  # Random weight for demonstration
                G.add_edge(i, j, weight=weight)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightgreen', font_size=8, font_weight="bold", arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Qubit Operations Visualization with Weighted Edges")
    plt.show()

def main():
    n_species = 4  # Number of species
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
    required_columns = ['Year', 'Species1', 'Species2', 'Species3', 'Species4']
    if not all(col in population_data.columns for col in required_columns):
        print(f"CSV file must contain the following columns: {required_columns}")
        return

    # Get the initial year and first 24 years of data
    initial_year = population_data['Year'].iloc[0]
    end_year = initial_year + 23
    population_data = population_data[population_data['Year'] <= end_year].reset_index(drop=True)
    
    # Normalize data
    scaler = MinMaxScaler()
    population_data.iloc[:, 1:] = scaler.fit_transform(population_data.iloc[:, 1:].astype(float))
    
    # Simulation loop
    losses = []
    all_outputs = []
    for epoch in range(200):
        epoch_loss = 0
        outputs = []
        for i in range(len(population_data) - 1):
            data = torch.tensor(population_data.drop(columns=['Year']).iloc[i].values).float().view(1, -1)
            target = torch.tensor(population_data.drop(columns=['Year']).iloc[i+1].values).float().view(1, -1)
            
            output = model(data)
            
            optimizer.zero_grad()
            loss = torch.mean((output - target)**2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            outputs.append(output.detach().numpy().flatten())
        
        losses.append(epoch_loss / len(population_data))
        all_outputs.append(outputs)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss / len(population_data)}')

    # Convert all_outputs to numpy array for plotting
    final_outputs = np.array(all_outputs[-1]).squeeze()
    final_outputs = scaler.inverse_transform(final_outputs.reshape(-1, n_species))

    # Analysis and comparison of results
    plt.figure(figsize=(12, 6))
    for i in range(1, n_species + 1):
        plt.plot(population_data['Year'], scaler.inverse_transform(population_data.drop(columns=['Year']).values)[:, i-1], label=f'Actual Species{i}', marker='o')
        plt.plot(population_data['Year'][1:], final_outputs[:, i-1], label=f'Simulated Species{i}', marker='x')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title('Population Projection for 24 Years')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the loss over epochs
    plt.figure(figsize=(12, 6))
    plt.plot(range(200), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize the qubits
    visualize_qubits(n_species)

if __name__ == '__main__':
    main()
