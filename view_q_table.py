import pickle
import numpy as np
import csv

# Load the Q-table from the .pkl file
def load_q_table(file_path):
    try:
        with open(file_path, 'rb') as file:
            q_table = pickle.load(file)
        print("Q-table successfully loaded.")
        return q_table
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return None

# Display basic information about the Q-table
def view_q_table(q_table):
    if q_table is not None:
        print("Q-Table Shape:", q_table.shape)
        print("Sample Values (first 5 states and actions):")
        print(q_table[:5, :5, :])  
    else:
        print("No Q-table data to display.")

# Save Q-table to a CSV file
def save_q_table_to_csv(q_table, output_csv):
    try:
        with open(output_csv, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["State_X", "State_Y", "Action_0", "Action_1", "Action_2", "Action_3"])  
            for i in range(q_table.shape[0]):
                for j in range(q_table.shape[1]):
                    writer.writerow([i, j, *q_table[i, j, :]])
        print(f"Q-table saved as {output_csv}")
    except Exception as e:
        print(f"Error saving Q-table to CSV: {e}")

# Main function
if __name__ == "__main__":
    # Path to the .pkl file
    file_path = 'lunar_lander.pkl'  

    # Load the Q-table
    q_table = load_q_table(file_path)

    # View the Q-table in the console
    view_q_table(q_table)

    
    output_csv = 'q_table.csv'
    save_q_table_to_csv(q_table, output_csv)
