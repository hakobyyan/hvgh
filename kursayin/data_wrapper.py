import pandas as pd
import re


def parse_data(text):
    # Split the text into separate tables based on the table headers
    tables = re.split(r'╒═══════╤═════════════════╤═══════╕', text)[1:]

    all_data = {}

    for table in tables:
        # Extract Z value from the first line
        z_match = re.search(r'Z=(\d+)', table)
        if not z_match:
            continue
        z_value = int(z_match.group(1))

        # Get the lines of data
        lines = table.strip().split('\n')[2:-1]  # Skip header and footer lines

        # Parse data into lists
        z_values = []
        varphi_values = []
        x_values = []

        for line in lines:
            try:
                parts = line.split('│')[1:-1]  # Remove outer borders
                if len(parts) != 3:
                    continue

                z = int(parts[0].strip())
                varphi = float(parts[1].strip())
                x = int(parts[2].strip())

                z_values.append(z)
                varphi_values.append(varphi)
                x_values.append(x)
            except (ValueError, IndexError) as e:
                print(f"Error processing line in Z={z_value}: '{line}' - {str(e)}")
                continue

        # Create DataFrame for this Z value
        if z_values:  # Only create DataFrame if we have data
            df = pd.DataFrame({
                f'Z={z_value}': z_values,
                f'varphi_{z_value}(z_{z_value})': varphi_values,
                f'X_{z_value}': x_values
            })
            all_data[z_value] = df

    return all_data


def write_to_excel(data_dict, output_file='output.xlsx'):
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for z_value, df in data_dict.items():
            sheet_name = f'Z={z_value}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).map(len).max(), len(col))
                worksheet.set_column(i, i, max_length + 2)


def main():
    try:
        with open('Distributor_tables.txt', 'r', encoding='utf-8') as file:
            text_data = file.read()

        parsed_data = parse_data(text_data)

        if not parsed_data:
            print("No data was successfully parsed from the file.")
            return

        write_to_excel(parsed_data, 'output.xlsx')
        print("Data has been successfully written to output.xlsx")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()