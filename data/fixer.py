"""
This script serves only as a reference for how the OFF files in the original
ModelNet40 dataset were cleaned - as they were, off2obj fails to parse some of
them, as the file format and vertex count were on the same line.
"""

import os

def fix(filename):
    # load file
    a = open(filename, "r")
    lines = a.readlines()
    a.close()

    # check if file malformed (first line has more than just "OFF")
    if lines[1].count(" ") == 2:
        # separate first line into two lines
        lines.insert(1, lines[0].split("OFF")[1])
        lines[0] = 'OFF\n'

    # save file
    a = open(filename, "w")
    a.writelines(lines)
    a.close()

def main():
    # walk through all files
    for root, dirs, files in os.walk(".", topdown = False):
        for name in files:

            # fix files ending in .off
            if name[-4::] == ".off":
                print(name)
                fix(os.path.join(root, name))

if __name__ == "__main__":
    main()
