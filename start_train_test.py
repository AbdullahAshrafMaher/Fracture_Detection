import argparse
import sys
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Data Postprocess')
    parser.add_argument('--model', type=str, default=None, help='load the model')
    parser.add_argument('--data_dir', type=str, default=None, help='the dir to data')
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='path to save training log')
    return parser.parse_args()

def main():
    args = parse_args()

    # Open the log file for writing
    log_file = open(args.log_file, 'w')

    # Redirect stdout and stderr to the log file
    sys.stdout = log_file
    sys.stderr = log_file

    # Initialize YOLO model
    model = YOLO(args.model)

    # Train the model
    model.train(data=args.data_dir)

    # Close the log file
    log_file.close()

if __name__ == '__main__':
    main()
