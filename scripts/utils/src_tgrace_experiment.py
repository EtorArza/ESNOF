import csv
import os

class ObjectiveLogger:
    def __init__(self, file_path, replace_existing=False, logevery=1):
        # Create the directory if it doesn't exist
        log_dir = os.path.dirname(file_path)
        self.logevery=logevery
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Check if the log file already exists
        if os.path.exists(file_path):
            if replace_existing:
                os.remove(file_path)
            else:
                raise FileExistsError(f"The log file '{file_path}' already exists.")

        self.file_path = file_path
        self.row_count = 0  # Initialize row count

        # Open the file and create the CSV writer
        self.csvfile = open(self.file_path, 'a', newline='')
        self.writer = csv.writer(self.csvfile)

        self.header_written = False


    def log_values(self, time, values):

        if not self.header_written:
            # Write header row
            self.writer.writerow(["time"] + list(range(len(values[::self.logevery]))))
            self.header_written = True


        # Increment row count and add it as the first value
        self.row_count += 1

        # # Round objective values to 3 decimals
        # rounded_values = [round(val, 3) for val in values]

        values = values[::self.logevery]

        # Write the row
        self.writer.writerow([time] + list(values))

    def close(self):
        # Close the CSV file
        self.csvfile.close()

def example_usage():
    # Example usage:
    log_file_path = 'results/data/tgrace_experiment/log.csv'

    # Initialize the logger with the log file path
    logger = ObjectiveLogger(log_file_path, replace_existing=True)

    # Log multiple rows of objective values (rounded to 3 decimals)
    logger.log_values(1.2, [1.12345, 2.56789, 3.98765])
    logger.log_values(1.4, [4.54321, 5.67890, 6.87654])

    # Close the logger to save the changes to the file
    logger.close()

    # Reading the CSV file using pandas
    import pandas as pd
    df = pd.read_csv(log_file_path)

    # Print the DataFrame
    print(df)


