import argparse
import glob

parser = argparse.ArgumentParser(description='Analyze submodules')
parser.add_argument('--dir', type=str, help='path to a pattern grouped directory')
args = parser.parse_args()

def get_contents(file):
    with open(file, "r") as reader:
        contents = reader.readlines()
    return contents

def get_target_lines(content, startstr, endstr):
    target_lines=[]
    start = False
    end = False
    for line in content:
        line = line.strip()
        if line.startswith(startstr):
            start = True
            continue
        if start and not end:
            if line.startswith(endstr):
                end = True
            if not end:
                target_lines.append(line)
        if end:
            return target_lines

def is_part_equivalent(c1, c2, desc):
    if desc == "init":
        startstr = "super(M, self).__init__()"
        endstr = "def forward"
    elif desc == "forward":
        startstr = "def forward"
        endstr = "m = M().eval()"
    elif desc == "inputs":
        startstr = "m = M().eval()"
        endstr = "start = time.time()"
    c1_part = get_target_lines(c1, startstr, endstr)
    c2_part = get_target_lines(c2, startstr, endstr)
    if len(c1_part) != len(c2_part):
        return False
    for i in range(len(c1_part)):
        if desc != "forward" and c1_part[i].split("=")[-1].strip() != c2_part[i].split("=")[-1].strip():
            return False
        if desc == "forward":
            c1_parts = c1_part[i].replace("=", " ").replace(",", " ").replace("(", " ").replace(")", " ").replace("  ", " ").split(" ")
            c2_parts = c2_part[i].replace("=", " ").replace(",", " ").replace("(", " ").replace(")", " ").replace("  ", " ").split(" ")
            if len(c1_parts) != len(c2_parts):
                return False
            else:
                for j in range(len(c1_parts)):
                    c1_parts[j] = c1_parts[j].strip()
                    c2_parts[j] = c2_parts[j].strip()
                    if c1_parts[j].startswith("x") and c2_parts[j].startswith("x"):
                        continue
                    if c1_parts[j].startswith("self.") and c2_parts[j].startswith("self."):
                        continue
                    if c1_parts[j] != c2_parts[j]:
                        return False
    return True

def is_equivalent(c1, c2):
    for desc in ["init", "forward", "inputs"]:
        if not is_part_equivalent(c1, c2, desc):
            return False
    return True

def deduplicate(idx, files, results):
    target_file = files[idx]
    results[target_file] = 1
    for j in range(idx+1, len(files)):
        if files[j] in results:
            continue
        target_contents = get_contents(target_file)
        cur_contents = get_contents(files[j])
        if is_equivalent(target_contents, cur_contents):
            results[target_file] += 1
            results[files[j]] = 0

dir = args.dir
files = glob.glob(dir+"/*.py")
results = {}
for i in range(len(files)):
    if files[i] in results:
        continue
    deduplicate(i, files, results)

with open(dir+"/all_deduplicated.txt", "w") as writer:
    for key in results:
        if results[key] > 0:
            writer.write(key+" "+str(results[key])+"\n")

with open(dir+"/all.txt", "w") as writer:
    for key in results:
        if results[key] > 0:
            writer.write(key+" "+str(results[key])+"\n")
        else:
            writer.write("**"+key+"\n")