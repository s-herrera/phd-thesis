import argparse

def remove_misc(input_file, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile, open(input_file, encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if line.startswith("# sent_id ="):
                line = line.strip() + f"{i}\n"
                outfile.write(line)
            elif line.strip() == "" or line.startswith("#"):
                outfile.write(line)
            else:
                columns = line.split("\t")
                if len(columns) >= 10:
                    columns[9] = "_"
                    columns[8] = "_" 
                outfile.write("\t".join(columns))
                outfile.write("\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Remove MISC column")
    parser.add_argument("-i", "--input_file", required=True, help="Path to the input file.")
    parser.add_argument("-o", "--output_file", required=True, help="Path to the output file.")
    args = parser.parse_args()

    remove_misc(args.input_file, args.output_file)