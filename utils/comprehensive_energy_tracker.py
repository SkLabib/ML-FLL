import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class EnergyTracker:
    """
    A comprehensive energy tracking system for federated learning.
    
    This class monitors CPU and RAM usage during training and calculates energy consumption
    metrics to evaluate the energy efficiency of the federated learning process.
    
    Attributes:
        cpu_tdp (float): Thermal Design Power of the CPU in watts.
        sampling_interval (float): Time interval between energy measurements in seconds.
        start_time (float): Timestamp when tracking started.
        measurements (list): List of energy measurements.
        client_measurements (dict): Dictionary of energy measurements per client.
        round_measurements (dict): Dictionary of energy measurements per round.
        carbon_intensity (float): Carbon intensity factor in kgCO2/kWh.
    """
    
    def __init__(self, cpu_tdp=65.0, sampling_interval=1.0, carbon_intensity=0.475):
        """
        Initialize the EnergyTracker.
        
        Args:
            cpu_tdp (float): Thermal Design Power of the CPU in watts.
            sampling_interval (float): Time interval between energy measurements in seconds.
            carbon_intensity (float): Carbon intensity factor in kgCO2/kWh (default: US average).
        """
        self.cpu_tdp = cpu_tdp
        self.sampling_interval = sampling_interval
        self.start_time = None
        self.measurements = []
        self.client_measurements = {}
        self.round_measurements = {}
        self.carbon_intensity = carbon_intensity
        self.total_energy = 0.0
        self.total_samples_processed = 0
        self.accuracy_history = []
        self.tracking_active = False
        self.tracking_start_time = None
        self.tracking_start_cpu = None
        self.tracking_start_memory = None
        self.round_energy_log = {}
        
    def start_tracking(self):
        """
        Start tracking energy usage for a specific operation.
        Records the starting state of CPU and memory usage.
        """
        self.tracking_active = True
        self.tracking_start_time = time.time()
        self.tracking_start_cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        self.tracking_start_memory = memory.used / (1024 ** 3)  # in GB
        return self.tracking_start_time
        
    def stop_tracking(self, client_id=None, round_num=None, operation_type="training"):
        """
        Stop tracking energy usage and calculate the energy consumed during the operation.
        
        Args:
            client_id (int, optional): ID of the client being measured.
            round_num (int, optional): Current federated learning round number.
            operation_type (str): Type of operation ("training" or "aggregation")
            
        Returns:
            dict: Energy metrics for the tracked operation.
        """
        if not self.tracking_active:
            return {"total_energy_wh": 0.0, "duration_seconds": 0.0}
            
        end_time = time.time()
        duration = end_time - self.tracking_start_time
        
        # Get current CPU and memory usage
        end_cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        end_memory = memory.used / (1024 ** 3)  # in GB
        
        # Calculate average CPU and memory usage during the operation
        avg_cpu = (self.tracking_start_cpu + end_cpu) / 2
        avg_memory = (self.tracking_start_memory + end_memory) / 2
        
        # Calculate energy consumption
        cpu_energy = (avg_cpu / 100.0) * self.cpu_tdp * duration / 3600  # in watt-hours
        ram_energy = avg_memory * 0.3 * duration / 3600  # Approximate RAM energy (0.3W per GB)
        total_energy = cpu_energy + ram_energy
        
        # Update total energy
        self.total_energy += total_energy
        
        # Calculate carbon footprint
        carbon_footprint = total_energy * self.carbon_intensity / 1000  # in kg CO2
        
        # Create energy metrics
        metrics = {
            "duration_seconds": duration,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_gb": avg_memory,
            "cpu_energy_wh": cpu_energy,
            "ram_energy_wh": ram_energy,
            "total_energy_wh": total_energy,
            "carbon_footprint_kg": carbon_footprint,
            "client_id": client_id,
            "round_num": round_num,
            "operation_type": operation_type
        }
        
        # Log round energy if round_num is provided
        if round_num is not None:
            if round_num not in self.round_energy_log:
                self.round_energy_log[round_num] = {
                    "Energy_Global": 0.0
                }
            
            if operation_type == "aggregation":
                self.round_energy_log[round_num]["Energy_Global"] = total_energy
            elif client_id is not None:
                self.round_energy_log[round_num][f"Energy_Client{client_id+1}"] = total_energy
        
        # Reset tracking state
        self.tracking_active = False
        
        return metrics
        
    def start(self):
        """
        Start energy tracking (legacy method, use start_tracking instead).
        """
        self.start_time = time.time()
        self.measurements = []
        
    def measure(self, client_id=None, round_num=None, samples_processed=0, accuracy=None):
        """
        Take a measurement of current energy usage.
        
        Args:
            client_id (int, optional): ID of the client being measured.
            round_num (int, optional): Current federated learning round number.
            samples_processed (int, optional): Number of samples processed in this measurement.
            accuracy (float, optional): Current model accuracy.
        
        Returns:
            dict: The measurement data.
        """
        if self.start_time is None:
            self.start()
            
        # Get current CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024 ** 3)  # Convert to GB
        
        # Calculate energy consumption
        elapsed_time = time.time() - self.start_time
        cpu_energy = (cpu_percent / 100.0) * self.cpu_tdp * self.sampling_interval / 3600  # in watt-hours
        ram_energy = memory_used_gb * 0.3 * self.sampling_interval / 3600  # Approximate RAM energy (0.3W per GB)
        total_energy = cpu_energy + ram_energy
        
        # Update total energy and samples
        self.total_energy += total_energy
        self.total_samples_processed += samples_processed
        
        # Calculate carbon footprint
        carbon_footprint = total_energy * self.carbon_intensity / 1000  # in kg CO2
        
        # Create measurement record
        measurement = {
            'timestamp': time.time(),
            'elapsed_time': elapsed_time,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'cpu_energy_wh': cpu_energy,
            'ram_energy_wh': ram_energy,
            'total_energy_wh': total_energy,
            'carbon_footprint_kg': carbon_footprint,
            'client_id': client_id,
            'round_num': round_num,
            'samples_processed': samples_processed
        }
        
        self.measurements.append(measurement)
        
        # Store client-specific measurements
        if client_id is not None:
            if client_id not in self.client_measurements:
                self.client_measurements[client_id] = []
            self.client_measurements[client_id].append(measurement)
        
        # Store round-specific measurements
        if round_num is not None:
            if round_num not in self.round_measurements:
                self.round_measurements[round_num] = []
            self.round_measurements[round_num].append(measurement)
        
        # Store accuracy if provided
        if accuracy is not None:
            self.accuracy_history.append((elapsed_time, total_energy, accuracy))
        
        return measurement
    
    def get_energy_per_sample(self):
        """
        Calculate the energy consumed per sample processed.
        
        Returns:
            float: Energy per sample in watt-hours.
        """
        if self.total_samples_processed == 0:
            return 0.0
        return self.total_energy / self.total_samples_processed
    
    def get_training_efficiency(self):
        """
        Calculate the training efficiency as (ΔAccuracy × Samples) / Energy_Consumed.
        
        Returns:
            float: Training efficiency metric.
        """
        if not self.accuracy_history or self.total_energy == 0:
            return 0.0
        
        initial_accuracy = self.accuracy_history[0][2] if self.accuracy_history else 0
        final_accuracy = self.accuracy_history[-1][2] if self.accuracy_history else 0
        accuracy_delta = final_accuracy - initial_accuracy
        
        return (accuracy_delta * self.total_samples_processed) / self.total_energy
    
    def get_client_energy_efficiency(self, client_id):
        """
        Calculate the energy efficiency for a specific client.
        
        Args:
            client_id (int): ID of the client.
            
        Returns:
            dict: Energy efficiency metrics for the client.
        """
        if client_id not in self.client_measurements or not self.client_measurements[client_id]:
            return {'energy_per_sample': 0.0, 'total_energy': 0.0, 'samples_processed': 0}
        
        client_data = self.client_measurements[client_id]
        total_energy = sum(m['total_energy_wh'] for m in client_data)
        samples_processed = sum(m['samples_processed'] for m in client_data)
        
        energy_per_sample = total_energy / samples_processed if samples_processed > 0 else 0.0
        
        return {
            'energy_per_sample': energy_per_sample,
            'total_energy': total_energy,
            'samples_processed': samples_processed
        }
    
    def get_round_energy(self, round_num):
        """
        Calculate the energy consumed in a specific round.
        
        Args:
            round_num (int): Round number.
            
        Returns:
            float: Total energy consumed in the round in watt-hours.
        """
        if round_num not in self.round_measurements:
            return 0.0
        
        return sum(m['total_energy_wh'] for m in self.round_measurements[round_num])
    
    def get_total_carbon_footprint(self):
        """
        Calculate the total carbon footprint of the training process.
        
        Returns:
            float: Total carbon footprint in kg CO2.
        """
        return self.total_energy * self.carbon_intensity / 1000
    
    def calculate_client_selection_score(self, client_id, accuracy, data_quality):
        """
        Calculate an energy-aware client selection score.
        
        Score = (Accuracy × Data_Quality) / Energy_Cost
        
        Args:
            client_id (int): ID of the client.
            accuracy (float): Client's model accuracy.
            data_quality (float): Quality score of client's data (0-1).
            
        Returns:
            float: Client selection score.
        """
        client_efficiency = self.get_client_energy_efficiency(client_id)
        energy_cost = client_efficiency['energy_per_sample']
        
        if energy_cost == 0:
            return 0.0
        
        return (accuracy * data_quality) / energy_cost
    
    def update_global_metrics(self, accuracy, energy_consumed, round_num):
        """
        Update global metrics with accuracy and energy consumption for a round.
        
        Args:
            accuracy (float): Global model accuracy for the round.
            energy_consumed (float or dict): Total energy consumed in the round.
                                           If dict, should contain 'total_energy_wh' key.
            round_num (int): Round number.
        """
        # Extract energy value from dict if needed
        if isinstance(energy_consumed, dict):
            energy_value = energy_consumed.get('total_energy_wh', 0.0)
        else:
            energy_value = float(energy_consumed)
        
        # Update accuracy history
        current_time = time.time()
        cumulative_energy = self.total_energy + energy_value
        self.accuracy_history.append((current_time, cumulative_energy, accuracy))
        
        # Update round energy log
        if round_num not in self.round_energy_log:
            self.round_energy_log[round_num] = {}
        
        self.round_energy_log[round_num]['global_accuracy'] = accuracy
        self.round_energy_log[round_num]['total_energy'] = energy_value
        
        # Update total energy
        self.total_energy += energy_value
    
    def save_measurements(self, filepath):
        """
        Save all measurements to a CSV file.
        
        Args:
            filepath (str): Path to save the CSV file.
        """
        df = pd.DataFrame(self.measurements)
        df.to_csv(filepath, index=False)
        
    def generate_report(self, output_dir='./energy_reports'):
        """
        Generate a comprehensive energy efficiency report with visualizations.
        
        Args:
            output_dir (str): Directory to save the report files.
            
        Returns:
            str: Path to the generated report directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(output_dir, f"energy_report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate a simple summary file
        summary_file = os.path.join(report_dir, "energy_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Energy Efficiency Summary Report\n")
            f.write("==============================\n\n")
            f.write(f"Total Energy Consumed: {self.total_energy:.4f} Wh\n")
            f.write(f"Total Samples Processed: {self.total_samples_processed}\n")
            f.write(f"Energy per Sample: {self.get_energy_per_sample():.6f} Wh\n")
            f.write(f"Training Efficiency: {self.get_training_efficiency():.6f}\n")
            f.write(f"Total Carbon Footprint: {self.get_total_carbon_footprint():.6f} kg CO2\n")
        
        return report_dir
        
    def generate_reports(self):
        """Generate energy efficiency reports."""
        # Save round energy log to CSV
        self.save_round_energy_log()
        return self.generate_report()
        
    def save_round_energy_log(self, filepath='./results/energy_log.csv'):
        """Save round energy log to a CSV file.
        
        Args:
            filepath (str): Path to save the CSV file.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert round energy log to DataFrame
        if not self.round_energy_log:
            print("No round energy data to save.")
            return
            
        # Create a DataFrame with Round column
        df = pd.DataFrame()
        df['Round'] = sorted(self.round_energy_log.keys())
        
        # Add client energy columns
        for round_num in sorted(self.round_energy_log.keys()):
            for key, value in self.round_energy_log[round_num].items():
                if key not in df.columns:
                    df[key] = 0.0
                df.loc[df['Round'] == round_num, key] = value
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Round energy log saved to {filepath}")


class ClientEnergyTracker:
    """
    Energy tracker for individual federated learning clients.
    
    This class is a lightweight wrapper around the main EnergyTracker that focuses
    on tracking energy consumption for a specific client.
    """
    
    def __init__(self, client_id, global_tracker):
        """
        Initialize a client-specific energy tracker.
        
        Args:
            client_id (int): ID of the client.
            global_tracker (EnergyTracker): Reference to the global energy tracker.
        """
        self.client_id = client_id
        self.global_tracker = global_tracker
        self.samples_processed = 0
        self.current_accuracy = 0.0
        self.current_round = None
        
    def start_tracking(self):
        """
        Start tracking energy for client training.
        
        Returns:
            float: Start timestamp.
        """
        return self.global_tracker.start_tracking()
        
    def stop_tracking(self, round_num=None):
        """
        Stop tracking energy and record measurements.
        
        Args:
            round_num (int, optional): Current federated learning round number.
            
        Returns:
            dict: Energy metrics for the tracked operation.
        """
        self.current_round = round_num
        return self.global_tracker.stop_tracking(
            client_id=self.client_id,
            round_num=round_num,
            operation_type="training"
        )
        
    def start_batch(self, batch_size):
        """
        Start tracking energy for a new batch (legacy method).
        
        Args:
            batch_size (int): Size of the batch being processed.
        """
        self.batch_size = batch_size
        self.batch_start_time = time.time()
        
    def end_batch(self, round_num=None):
        """
        End tracking for the current batch and record measurements (legacy method).
        
        Args:
            round_num (int, optional): Current federated learning round number.
            
        Returns:
            dict: The measurement data.
        """
        self.samples_processed += self.batch_size
        self.current_round = round_num
        return self.global_tracker.measure(
            client_id=self.client_id,
            round_num=round_num,
            samples_processed=self.batch_size,
            accuracy=self.current_accuracy
        )
    
    def update_accuracy(self, accuracy):
        """
        Update the current accuracy of the client's model.
        
        Args:
            accuracy (float): New accuracy value.
        """
        self.current_accuracy = accuracy
        
    def get_efficiency_metrics(self):
        """
        Get energy efficiency metrics for this client.
        
        Returns:
            dict: Energy efficiency metrics.
        """
        return self.global_tracker.get_client_energy_efficiency(self.client_id)
    
    def get_selection_score(self, data_quality):
        """
        Calculate the client selection score based on accuracy, data quality, and energy efficiency.
        
        Args:
            data_quality (float): Quality score of client's data (0-1).
            
        Returns:
            float: Client selection score.
        """
        return self.global_tracker.calculate_client_selection_score(
            self.client_id, self.current_accuracy, data_quality
        )
    
    def get_metrics(self):
        """
        Get energy metrics for this client.
        
        Returns:
            dict: Energy metrics including total_energy and training_efficiency.
        """
        efficiency_metrics = self.get_efficiency_metrics()
        return {
            'total_energy': efficiency_metrics.get('total_energy', 0),
            'training_efficiency': efficiency_metrics.get('energy_per_sample', 0)
        }