import json
import csv
import sys

def convert_test_cases(json_path, csv_path):
    """Convert test cases from JSON to CSV format, filtering for failed tests."""
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Prepare rows for CSV
    rows = []
    # If the input is a list of test cases directly
    test_cases = data['test_cases'] if 'test_cases' in data else data
    
    for test in test_cases:
        # Only include failed test cases
        if test.get('passed') is False:  # Specifically check for False
            row = {
                'id': test['id'].lstrip('q'),
                'prompt': test['input'],
                'A': test['metadata']['options']['A'],
                'B': test['metadata']['options']['B'],
                'C': test['metadata']['options']['C'],
                'D': test['metadata']['options']['D'],
                'E': test['metadata']['options']['E'],
                'actual': test['actual_output']  # Keep the original column name
            }
            rows.append(row)

    # Create new rows with renamed column
    renamed_rows = []
    for row in rows:
        new_row = row.copy()
        new_row['answer'] = new_row.pop('actual')  # Rename 'actual' to 'answer'
        renamed_rows.append(new_row)

    # Write CSV file with renamed column
    fieldnames = ['id', 'prompt', 'A', 'B', 'C', 'D', 'E', 'answer']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(renamed_rows)
        print(f"Successfully wrote {len(renamed_rows)} failed test cases to {csv_path}")

def format_math_csv(csv_path):
    """Read and format the math CSV file."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"\nProcessed {len(rows)} failed questions:")
        for row in rows:
            print(f"\nQuestion {row['id']}:")
            print(f"Prompt: {row['prompt']}")
            print("\nOptions:")
            print(f"A) {row['A']}")
            print(f"B) {row['B']}")
            print(f"C) {row['C']}")
            print(f"D) {row['D']}")
            print(f"E) {row['E']}")
            print(f"\nAnswer: {row['answer']}")
            print("-" * 80)
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_test_cases.py input.json output.csv")
        sys.exit(1)

    # Convert JSON to CSV with filtering
    convert_test_cases(sys.argv[1], sys.argv[2])
    
    # Format and display the CSV contents
    format_math_csv(sys.argv[2])
