
all_patterns=[]
for i in range(1, 11):
    f = "nnc_summary/nnc_summary_"+str(i)+".log"
    print(f)
    with open(f, "r") as reader:
        contents = reader.readlines()
        for line in contents:
            if "Pattern" in line:
                continue
            ops = line.strip().split(" ")[0].split(",")
            opstr = "("
            for op in ops:
                if op.strip() == "":
                    continue
                op = op.replace("max_pool2d", "maxpool2d")
                op = op.replace("avg_pool2d", "avgpool2d")
                op = op.replace("batch_norm", "batchnorm")
                op = op.replace("layer_norm", "layernorm")
                op = op.replace("adaptive_avg_pool2d", "adaptiveavgpool2d")
                opstr = opstr + "'"+op+"', "
            opstr = opstr+")"
            all_patterns.append(opstr)

with open("patterns.txt", "w") as f:
    for opstr in all_patterns:
        f.write(opstr+"\n")