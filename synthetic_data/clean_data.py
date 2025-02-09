import os
import json
from argparse import ArgumentParser

def clean_json_data(json_folder):
    """
    Removes JSON files from the specified folder that:
    1. Contain any of the following values in any field: [], 'null', '', ['null'], or "content": [null].
    2. Do not have exactly two fields.
    """
    
    bad_values = [[], 'null', '', '[null]', ['null']]
    
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(json_folder, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Check if the file has exactly two fields
                if len(data.keys()) != 2:
                    os.remove(filepath)
                    print(f"Removed {filename} because it does not have exactly two fields.")
                    continue  # Skip to the next file
                
                # Check if any value in the dictionary is a bad value
                if any(value in bad_values for value in data.values()):
                    os.remove(filepath)
                    print(f"Removed {filename} due to bad values.")
                    continue  # Skip to the next file

                # Check for "content": [null]
                if "content" in data and data["content"] == [None]:
                    os.remove(filepath)
                    print(f"Removed {filename} due to content: [null].")
                    continue  # Skip to the next file

            except json.JSONDecodeError:
                print(f"Could not read {filename}")
                os.remove(filepath)
                print(f"Removed {filename} due to decode error.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--json_folder", type=str, default="synthetic_data/json_files", help="Path to the folder containing JSON files")
    args = parser.parse_args()
    clean_json_data(args.json_folder)
