import logging
from tqdm import tqdm
import os
class JobLogger:
    def __init__(self, file_path, api=False):
        self.api = api
        self.file_path = file_path
        self.error_logger = logging.getLogger('error_logger')
        self.output_logger = logging.getLogger('output_logger')
        self.configure_loggers(api)

    
    def configure_loggers(self, api):
        logging.basicConfig(level=logging.INFO)
        
        # Create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path))
        
        if api:
            # Create file handlers for both types of logs
            error_file_handler = logging.FileHandler(f'{self.file_path}/errors.err', mode='w+')
            output_file_handler = logging.FileHandler(f'{self.file_path}/outputs.out', mode='w+')

            # Optionally, set the logging level for each handler
            error_file_handler.setLevel(logging.ERROR)
            output_file_handler.setLevel(logging.INFO)

            # Add the handlers to the loggers
            self.error_logger.addHandler(error_file_handler)
            self.output_logger.addHandler(output_file_handler)

    
    def log_error(self, message):
        self.error_logger.error(message)
    
    def log_output(self, message):
        if self.api:
            self.output_logger.info(message)
        else:
            print(message)

    
    def tqdm(self, *args, **kwargs):
        kwargs['file'] = open(f'{self.file_path}/progress.out', 'a')
        return tqdm(*args, **kwargs)

    
    def catch_errors(self, func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.log_error(f"An error occurred: {e}")
                # return e
        return wrapper
    
    def log_groupby_counts(self, df, groupby_cols):
        # Group the DataFrame and calculate counts
        grouped_df = df.groupby(groupby_cols).size().reset_index(name='counts')
        
        grouped_df = grouped_df.sort_values(by='num_shots')

        # Log the grouped DataFrame to a CSV file
        output_csv_path = os.path.join(self.file_path, 'grouped_counts.csv')
        grouped_df.to_markdown(output_csv_path, index=False)

        # Log a message to indicate successful saving of the DataFrame
        self.log_output(f"Counts data saved to {output_csv_path}")

        return grouped_df
