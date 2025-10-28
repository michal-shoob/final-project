from pyboolnet import file_exchange
import networkx as nx
from itertools import product

# Define the network in BNET format
bnet = """
v1,    1
v2,    v2
v3,    v1 | v4
v4,    v6 & v2
v5,    v1 & v2 & v3
v6,    v4 & v5
v7,    v5 | (v6 & v4)
"""

# Convert the network to primes format
primes = file_exchange.bnet2primes(bnet)

# List of nodes
nodes = list(primes.keys())

initial_values = {
    "v1": None,
    "v2": 1,
    "v3": 1,
    "v4": None,
    "v5": 1,
    "v6": 1,
    "v7": None

}

# Define target values
target_values = {
    "v1": 1,
    "v2": 1,
    "v3": 1,
    "v4": 1,
    "v5": 1,
    "v6": 1,
    "v7": 1
}

# Define edge functions (to keep them for completeness)
edge_functions = {
    ("v1", "v3"): "v1 | v4",
    ("v4", "v3"): "v6 & v2",
    ("v1", "v5"): "v1 & v2 & v3",
    ("v2", "v5"): "v1 & v2 & v3",
    ("v3", "v5"): "v1 & v2 & v3",
    ("v4", "v6"): "v4 & v5",
    ("v5", "v6"): "v4 & v5",
    ("v5", "v7"): "v5 | (v6 & v4)",
    ("v6", "v7"): "v5 | (v6 & v4)",
    ("v2", "v6"): "v2 | v4",
    ("v3", "v7"): "v3 & v5"
}

bnet_20 = """
v1,    1
v2,    v2
v3,    v1 | v4
v4,    v6 & v2
v5,    v1 & v2 & v3
v6,    v4 & v5
v7,    v5 | (v6 & v4)
v8,    v3 | v7
v9,    v2 & v8
v10,   v9 | (v1 & v6)
v11,   v5 & v8
v12,   v10 | v11
v13,   v12 & v6
v14,   v4 | v13
v15,   v14 & v9
v16,   v15 | v2
v17,   v13 & v7
v18,   v16 | (v11 & v3)
v19,   v17 & v18
v20,   v19 | v1
"""

# Convert the network to primes format
primes_20 = file_exchange.bnet2primes(bnet_20)

# List of nodes
nodes_20 = list(primes_20.keys())

initial_values_20 = {
    "v1": None,
    "v2": 1,
    "v3": 1,
    "v4": None,
    "v5": 1,
    "v6": 1,
    "v7": 0,
    "v8": None,
    "v9": None,
    "v10": None,
    "v11": None,
    "v12": None,
    "v13": None,
    "v14": None,
    "v15": None,
    "v16": None,
    "v17": None,
    "v18": None,
    "v19": None,
    "v20": None
}
target_values_20 = {
    "v1": 1,
    "v2": 1,
    "v3": 1,
    "v4": 1,
    "v5": 1,
    "v6": 1,
    "v7": 1,
    "v8": 1,
    "v9": 1,
    "v10": 1,
    "v11": 1,
    "v12": 1,
    "v13": 1,
    "v14": 1,
    "v15": 1,
    "v16": 1,
    "v17": 1,
    "v18": 1,
    "v19": 1,
    "v20": 1
}

edge_functions_20 = {
    ("v1", "v3"): "v1 | v4",
    ("v4", "v3"): "v1 | v4",

    ("v6", "v4"): "v6 & v2",
    ("v2", "v4"): "v6 & v2",

    ("v1", "v5"): "v1 & v2 & v3",
    ("v2", "v5"): "v1 & v2 & v3",
    ("v3", "v5"): "v1 & v2 & v3",

    ("v4", "v6"): "v4 & v5",
    ("v5", "v6"): "v4 & v5",

    ("v5", "v7"): "v5 | (v6 & v4)",
    ("v6", "v7"): "v5 | (v6 & v4)",
    ("v4", "v7"): "v5 | (v6 & v4)",

    ("v3", "v8"): "v3 | v7",
    ("v7", "v8"): "v3 | v7",

    ("v2", "v9"): "v2 & v8",
    ("v8", "v9"): "v2 & v8",

    ("v9", "v10"): "v9 | (v1 & v6)",
    ("v1", "v10"): "v9 | (v1 & v6)",
    ("v6", "v10"): "v9 | (v1 & v6)",

    ("v5", "v11"): "v5 & v8",
    ("v8", "v11"): "v5 & v8",

    ("v10", "v12"): "v10 | v11",
    ("v11", "v12"): "v10 | v11",

    ("v12", "v13"): "v12 & v6",
    ("v6", "v13"): "v12 & v6",

    ("v4", "v14"): "v4 | v13",
    ("v13", "v14"): "v4 | v13",

    ("v14", "v15"): "v14 & v9",
    ("v9", "v15"): "v14 & v9",

    ("v15", "v16"): "v15 | v2",
    ("v2", "v16"): "v15 | v2",

    ("v13", "v17"): "v13 & v7",
    ("v7", "v17"): "v13 & v7",

    ("v16", "v18"): "v16 | (v11 & v3)",
    ("v11", "v18"): "v16 | (v11 & v3)",
    ("v3", "v18"): "v16 | (v11 & v3)",

    ("v17", "v19"): "v17 & v18",
    ("v18", "v19"): "v17 & v18",

    ("v19", "v20"): "v19 | v1",
    ("v1", "v20"): "v19 | v1"
}

bnet_50 = """
v1,    1
v2,    v2
v3,    v1 | v4
v4,    v6 & v2
v5,    v1 & v2 & v3
v6,    v4 & v5
v7,    v5 | (v6 & v4)
v8,    v3 | v7
v9,    v2 & v8
v10,   v9 | (v1 & v6)
v11,   v5 & v8
v12,   v10 | v11
v13,   v12 & v6
v14,   v4 | v13
v15,   v14 & v9
v16,   v15 | v2
v17,   v13 & v7
v18,   v16 | (v11 & v3)
v19,   v17 & v18
v20,   v19 | v1
v21,   v20 & v2
v22,   v21 | v3
v23,   v22 & v4
v24,   v23 | v5
v25,   v24 & v6
v26,   v25 | v7
v27,   v26 & v8
v28,   v27 | v9
v29,   v28 & v10
v30,   v29 | v11
v31,   v30 & v12
v32,   v31 | v13
v33,   v32 & v14
v34,   v33 | v15
v35,   v34 & v16
v36,   v35 | v17
v37,   v36 & v18
v38,   v37 | v19
v39,   v38 & v20
v40,   v39 | v21
v41,   v40 & v22
v42,   v41 | v23
v43,   v42 & v24
v44,   v43 | v25
v45,   v44 & v26
v46,   v45 | v27
v47,   v46 & v28
v48,   v47 | v29
v49,   v48 & v30
v50,   v49 | v31
"""


# Convert the network to primes format
primes_50 = file_exchange.bnet2primes(bnet_50)

# List of nodes
nodes_50 = list(primes_50.keys())
initial_values_3 = {
    "v1": 1,
    "v2": 1,
    "v3": 0,
    "v4": 0,
    "v5": 0,
    "v6": 0,
    "v7": 0,
    "v8": 1,
    "v9": 1,
    "v10": 1,
    "v11": 1,
    "v12": 1,
    "v13": None,
    "v14": None,
    "v15": None,
    "v16": None,
    "v17": None,
    "v18": 1,
    "v19": 1,
    "v20": 1,
    "v21": 1,
    "v22": 1,
    "v23": 1,
    "v24": 1,
    "v25": 1,
    "v26": 1,
    "v27": 1,
    "v28": 1,
    "v29": 1,
    "v30": 1,
    "v31": 1,
    "v32": 1,
    "v33": 1,
    "v34": 1,
    "v35": 1,
    "v36": 1,
    "v37": 1,
    "v38": 1,
    "v39": 1,
    "v40": 1,
    "v41": 1,
    "v42": 1,
    "v43": 1,
    "v44": 1,
    "v45": 1,
    "v46": 1,
    "v47": 1,
    "v48": 1,
    "v49": 1,
    "v50": 1
}

initial_values_50 = {
    "v1": 1,
    "v2": None,
    "v3": 0,
    "v4": 0,
    "v5": None,
    "v6": None,
    "v7": None,
    "v8": 1,
    "v9": 1,
    "v10": 1,
    "v11": 1,
    "v12": 1,
    "v13": None,
    "v14": None,
    "v15": None,
    "v16": None,
    "v17": None,
    "v18": 1,
    "v19": 1,
    "v20": None,
    "v21": None,
    "v22": 1,
    "v23": None,
    "v24": 1,
    "v25": 1,
    "v26": 1,
    "v27": None,
    "v28": 1,
    "v29": 1,
    "v30": None,
    "v31": None,
    "v32": 1,
    "v33": None,
    "v34": None,
    "v35": 1,
    "v36": 1,
    "v37": None,
    "v38": 1,
    "v39": None,
    "v40": 1,
    "v41": 1,
    "v42": None,
    "v43": 1,
    "v44": 1,
    "v45": None,
    "v46": 1,
    "v47": 1,
    "v48": None,
    "v49": 1,
    "v50": None
}

# Define target values
target_values_50 = {
    "v1": 1, "v2": 0, "v3": 0, "v4": 0, "v5": 0,
    "v6": 0, "v7": 0, "v8": 1, "v9": 1, "v10": 1,
    "v11": 1, "v12": 1, "v13": 0, "v14": 0, "v15": 0,
    "v16": 0, "v17": 0, "v18": 1, "v19": 1, "v20": 1,
    "v21": 0, "v22": 1, "v23": 0, "v24": 1, "v25": 1,
    "v26": 1, "v27": 0, "v28": 1, "v29": 1, "v30": 0,
    "v31": 0, "v32": 1, "v33": 0, "v34": 0, "v35": 1,
    "v36": 1, "v37": 0, "v38": 1, "v39": 0, "v40": 1,
    "v41": 1, "v42": 0, "v43": 1, "v44": 1, "v45": 0,
    "v46": 1, "v47": 1, "v48": 0, "v49": 1, "v50": 0
}


# Define edge functions (to keep them for completeness)
edge_functions_50 = {
    ("v1", "v3"): "v1 | v4",
    ("v4", "v3"): "v1 | v4",

    ("v6", "v4"): "v6 & v2",
    ("v2", "v4"): "v6 & v2",

    ("v1", "v5"): "v1 & v2 & v3",
    ("v2", "v5"): "v1 & v2 & v3",
    ("v3", "v5"): "v1 & v2 & v3",

    ("v4", "v6"): "v4 & v5",
    ("v5", "v6"): "v4 & v5",

    ("v5", "v7"): "v5 | (v6 & v4)",
    ("v6", "v7"): "v5 | (v6 & v4)",
    ("v4", "v7"): "v5 | (v6 & v4)",

    ("v3", "v8"): "v3 | v7",
    ("v7", "v8"): "v3 | v7",

    ("v2", "v9"): "v2 & v8",
    ("v8", "v9"): "v2 & v8",

    ("v9", "v10"): "v9 | (v1 & v6)",
    ("v1", "v10"): "v9 | (v1 & v6)",
    ("v6", "v10"): "v9 | (v1 & v6)",

    ("v5", "v11"): "v5 & v8",
    ("v8", "v11"): "v5 & v8",

    ("v10", "v12"): "v10 | v11",
    ("v11", "v12"): "v10 | v11",

    ("v12", "v13"): "v12 & v6",
    ("v6", "v13"): "v12 & v6",

    ("v4", "v14"): "v4 | v13",
    ("v13", "v14"): "v4 | v13",

    ("v14", "v15"): "v14 & v9",
    ("v9", "v15"): "v14 & v9",

    ("v15", "v16"): "v15 | v2",
    ("v2", "v16"): "v15 | v2",

    ("v13", "v17"): "v13 & v7",
    ("v7", "v17"): "v13 & v7",

    ("v16", "v18"): "v16 | (v11 & v3)",
    ("v11", "v18"): "v16 | (v11 & v3)",
    ("v3", "v18"): "v16 | (v11 & v3)",

    ("v17", "v19"): "v17 & v18",
    ("v18", "v19"): "v17 & v18",

    ("v19", "v20"): "v19 | v1",
    ("v1", "v20"): "v19 | v1"
}

bnet_100 = """
v1,    1
v2,    v2
v3,    v1 | v4
v4,    v6 & v2
v5,    v1 & v2 & v3
v6,    v4 & v5
v7,    v5 | (v6 & v4)
v8,    v3 | v7
v9,    v2 & v8
v10,   v9 | (v1 & v6)
v11,   v5 & v8
v12,   v10 | v11
v13,   v12 & v6
v14,   v4 | v13
v15,   v14 & v9
v16,   v15 | v2
v17,   v13 & v7
v18,   v16 | (v11 & v3)
v19,   v17 & v18
v20,   v19 | v1
v21,   v20 & v7
v22,   v21 | v10
v23,   v22 & v14
v24,   v23 | v8
v25,   v24 & v16
v26,   v25 | (v12 & v3)
v27,   v26 & v20
v28,   v27 | v5
v29,   v28 & v22
v30,   v29 | v13
v31,   v30 & v24
v32,   v31 | (v18 & v7)
v33,   v32 & v26
v34,   v33 | v9
v35,   v34 & v28
v36,   v35 | v15
v37,   v36 & v30
v38,   v37 | (v21 & v11)
v39,   v38 & v32
v40,   v39 | v17
v41,   v40 & v34
v42,   v41 | v19
v43,   v42 & v36
v44,   v43 | (v25 & v9)
v45,   v44 & v37
v46,   v45 | v23
v47,   v46 & v40
v48,   v47 | v27
v49,   v48 & v41
v50,   v49 | (v31 & v13)
v51,   v50 & v44
v52,   v51 | v33
v53,   v52 & v46
v54,   v53 | (v35 & v20)
v55,   v54 & v48
v56,   v55 | v39
v57,   v56 & v50
v58,   v57 | v42
v59,   v58 & v52
v60,   v59 | (v44 & v24)
v61,   v60 & v54
v62,   v61 | v47
v63,   v62 & v56
v64,   v63 | v49
v65,   v64 & v58
v66,   v65 | (v52 & v28)
v67,   v66 & v60
v68,   v67 | v53
v69,   v68 & v62
v70,   v69 | v55
v71,   v70 & v64
v72,   v71 | (v58 & v34)
v73,   v72 & v66
v74,   v73 | v57
v75,   v74 & v68
v76,   v75 | v61
v77,   v76 & v70
v78,   v77 | (v63 & v41)
v79,   v78 & v72
v80,   v79 | v65
v81,   v80 & v74
v82,   v81 | v67
v83,   v82 & v76
v84,   v83 | (v69 & v45)
v85,   v84 & v78
v86,   v85 | v71
v87,   v86 & v80
v88,   v87 | v73
v89,   v88 & v82
v90,   v89 | (v75 & v51)
v91,   v90 & v84
v92,   v91 | v77
v93,   v92 & v86
v94,   v93 | v79
v95,   v94 & v88
v96,   v95 | (v81 & v59)
v97,   v96 & v90
v98,   v97 | v83
v99,   v98 & v92
v100,  v99 | v85
"""

primes_100 = file_exchange.bnet2primes(bnet_100)

nodes_100 = list(primes_100.keys())

initial_values_100 = {
    "v1": None,   "v2": None,   "v3": None, "v4": None,   "v5": None,
    "v6": 0,   "v7": 1,   "v8": None, "v9": 0,   "v10": None,
    "v11": 1,  "v12": 0,  "v13": None, "v14": 1,  "v15": None,
    "v16":None,  "v17": 1,  "v18": None, "v19": 0,  "v20": None,
    "v21": 1,  "v22": 0,  "v23": None, "v24": 1,  "v25": None,
    "v26": 0,  "v27": 1,  "v28": None, "v29": 0,  "v30": None,
    "v31": 1,  "v32": 0,  "v33": None, "v34": 1,  "v35": None,
    "v36": None,  "v37": 1,  "v38": None, "v39": 0,  "v40": None,
    "v41": 1,  "v42": 0,  "v43": None, "v44": 1,  "v45": None,
    "v46": 0,  "v47": 1,  "v48": None, "v49": 0,  "v50": None,
    "v51": 1,  "v52": 0,  "v53": None, "v54": 1,  "v55": None,
    "v56": None,  "v57": 1,  "v58": None, "v59": 0,  "v60": None,
    "v61": 1,  "v62": 0,  "v63": None, "v64": 1,  "v65": None,
    "v66": 0,  "v67": 1,  "v68": None, "v69": 0,  "v70": None,
    "v71": 1,  "v72": None,  "v73": None, "v74": 1,  "v75": None,
    "v76": 0,  "v77": 1,  "v78": None, "v79": 0,  "v80": None,
    "v81": 1,  "v82": None,  "v83": None, "v84": 1,  "v85": None,
    "v86": 0,  "v87": 1,  "v88": None, "v89": 0,  "v90": None,
    "v91": 1,  "v92": 0,  "v93": None, "v94": 1,  "v95": None,
    "v96": None,  "v97": 1,  "v98": None, "v99": 0,  "v100": None
}

target_values_100 = {
    "v1": 1, "v2": 1, "v3": 1, "v4": 1, "v5": 1,
    "v6": 1, "v7": 1, "v8": 1, "v9": 1, "v10": 1,
    "v11": 1, "v12": 1, "v13": 1, "v14": 1, "v15": 1,
    "v16": 1, "v17": 1, "v18": 1, "v19": 1, "v20": 1,
    "v21": 1, "v22": 1, "v23": 1, "v24": 1, "v25": 1,
    "v26": 1, "v27": 1, "v28": 1, "v29": 1, "v30": 1,
    "v31": 1, "v32": 1, "v33": 1, "v34": 1, "v35": 1,
    "v36": 1, "v37": 1, "v38": 1, "v39": 1, "v40": 1,
    "v41": 1, "v42": 1, "v43": 1, "v44": 1, "v45": 1,
    "v46": 1, "v47": 1, "v48": 1, "v49": 1, "v50": 1,
    "v51": 1, "v52": 1, "v53": 1, "v54": 1, "v55": 1,
    "v56": 1, "v57": 1, "v58": 1, "v59": 1, "v60": 1,
    "v61": 1, "v62": 1, "v63": 1, "v64": 1, "v65": 1,
    "v66": 1, "v67": 1, "v68": 1, "v69": 1, "v70": 1,
    "v71": 1, "v72": 1, "v73": 1, "v74": 1, "v75": 1,
    "v76": 1, "v77": 1, "v78": 1, "v79": 1, "v80": 1,
    "v81": 1, "v82": 1, "v83": 1, "v84": 1, "v85": 1,
    "v86": 1, "v87": 1, "v88": 1, "v89": 1, "v90": 1,
    "v91": 1, "v92": 1, "v93": 1, "v94": 1, "v95": 1,
    "v96": 1, "v97": 1, "v98": 1, "v99": 1, "v100": 1
}

edge_functions_100 = {
    ("v1", "v3"): "v1 | v4",
    ("v4", "v3"): "v1 | v4",
    ("v1", "v5"): "v1 & v2 & v3",
    ("v2", "v5"): "v1 & v2 & v3",
    ("v3", "v5"): "v1 & v2 & v3",
    ("v4", "v6"): "v4 & v5",
    ("v5", "v6"): "v4 & v5",
    ("v2", "v6"): "v2 | v4",
    ("v5", "v7"): "v5 | (v6 & v4)",
    ("v6", "v7"): "v5 | (v6 & v4)",
    ("v3", "v7"): "v3 & v5",
    ("v3", "v8"): "v3 | v7",
    ("v7", "v8"): "v3 | v7",
    ("v2", "v9"): "v2 & v8",
    ("v8", "v9"): "v2 & v8",
    ("v9", "v10"): "v9 | (v1 & v6)",
    ("v1", "v10"): "v9 | (v1 & v6)",
    ("v6", "v10"): "v9 | (v1 & v6)",
    ("v5", "v11"): "v5 & v8",
    ("v8", "v11"): "v5 & v8",
    ("v10", "v12"): "v10 | v11",
    ("v11", "v12"): "v10 | v11",
    ("v12", "v13"): "v12 & v6",
    ("v6", "v13"): "v12 & v6",
    ("v4", "v14"): "v4 | v13",
    ("v13", "v14"): "v4 | v13",
    ("v14", "v15"): "v14 & v9",
    ("v9", "v15"): "v14 & v9",
    ("v15", "v16"): "v15 | v2",
    ("v2", "v16"): "v15 | v2",
    ("v13", "v17"): "v13 & v7",
    ("v7", "v17"): "v13 & v7",
    ("v16", "v18"): "v16 | (v11 & v3)",
    ("v11", "v18"): "v16 | (v11 & v3)",
    ("v3", "v18"): "v16 | (v11 & v3)",
    ("v17", "v19"): "v17 & v18",
    ("v18", "v19"): "v17 & v18",
    ("v19", "v20"): "v19 | v1",
    ("v1", "v20"): "v19 | v1",
    ("v20", "v21"): "v20 & v7",
    ("v7", "v21"): "v20 & v7",
    ("v21", "v22"): "v21 | v10",
    ("v10", "v22"): "v21 | v10",
    ("v22", "v23"): "v22 & v14",
    ("v14", "v23"): "v22 & v14",
    ("v23", "v24"): "v23 | v8",
    ("v8", "v24"): "v23 | v8",
    ("v24", "v25"): "v24 & v16",
    ("v16", "v25"): "v24 & v16",
    ("v25", "v26"): "v25 | (v12 & v3)",
    ("v12", "v26"): "v25 | (v12 & v3)",
    ("v3", "v26"): "v25 | (v12 & v3)",
    ("v26", "v27"): "v26 & v20",
    ("v20", "v27"): "v26 & v20",
    ("v27", "v28"): "v27 | v5",
    ("v5", "v28"): "v27 | v5",
    ("v28", "v29"): "v28 & v22",
    ("v22", "v29"): "v28 & v22",
    ("v29", "v30"): "v29 | v13",
    ("v13", "v30"): "v29 | v13",
    ("v30", "v31"): "v30 & v24",
    ("v24", "v31"): "v30 & v24",
    ("v31", "v32"): "v31 | (v18 & v7)",
    ("v18", "v32"): "v31 | (v18 & v7)",
    ("v7", "v32"): "v31 | (v18 & v7)",
    ("v32", "v33"): "v32 & v26",
    ("v26", "v33"): "v32 & v26",
    ("v33", "v34"): "v33 | v9",
    ("v9", "v34"): "v33 | v9",
    ("v34", "v35"): "v34 & v28",
    ("v28", "v35"): "v34 & v28",
    ("v35", "v36"): "v35 | v15",
    ("v15", "v36"): "v35 | v15",
    ("v36", "v37"): "v36 & v30",
    ("v30", "v37"): "v36 & v30",
    ("v37", "v38"): "v37 | (v21 & v11)",
    ("v21", "v38"): "v37 | (v21 & v11)",
    ("v11", "v38"): "v37 | (v21 & v11)",
    ("v38", "v39"): "v38 & v32",
    ("v32", "v39"): "v38 & v32",
    ("v39", "v40"): "v39 | v17",
    ("v17", "v40"): "v39 | v17",
    ("v40", "v41"): "v40 & v34",
    ("v34", "v41"): "v40 & v34",
    ("v41", "v42"): "v41 | v19",
    ("v19", "v42"): "v41 | v19",
    ("v42", "v43"): "v42 & v36",
    ("v36", "v43"): "v42 & v36",
    ("v43", "v44"): "v43 | (v25 & v9)",
    ("v25", "v44"): "v43 | (v25 & v9)",
    ("v9", "v44"): "v43 | (v25 & v9)",
    ("v44", "v45"): "v44 & v37",
    ("v37", "v45"): "v44 & v37",
    ("v45", "v46"): "v45 | v23",
    ("v23", "v46"): "v45 | v23",
    ("v46", "v47"): "v46 & v40",
    ("v40", "v47"): "v46 & v40",
    ("v47", "v48"): "v47 | v27",
    ("v27", "v48"): "v47 | v27",
    ("v48", "v49"): "v48 & v41",
    ("v41", "v49"): "v48 & v41",
    ("v49", "v50"): "v49 | (v31 & v13)",
    ("v31", "v50"): "v49 | (v31 & v13)",
    ("v13", "v50"): "v49 | (v31 & v13)",
    ("v50", "v51"): "v50 & v44",
    ("v44", "v51"): "v50 & v44",
    ("v51", "v52"): "v51 | v33",
    ("v33", "v52"): "v51 | v33",
    ("v52", "v53"): "v52 & v46",
    ("v46", "v53"): "v52 & v46",
    ("v53", "v54"): "v53 | (v35 & v20)",
    ("v35", "v54"): "v53 | (v35 & v20)",
    ("v20", "v54"): "v53 | (v35 & v20)",
    ("v54", "v55"): "v54 & v48",
    ("v48", "v55"): "v54 & v48",
    ("v55", "v56"): "v55 | v39",
    ("v39", "v56"): "v55 | v39",
    ("v56", "v57"): "v56 & v50",
    ("v50", "v57"): "v56 & v50",
    ("v57", "v58"): "v57 | v42",
    ("v42", "v58"): "v57 | v42",
    ("v58", "v59"): "v58 & v52",
    ("v52", "v59"): "v58 & v52",
    ("v59", "v60"): "v59 | (v44 & v24)",
    ("v44", "v60"): "v59 | (v44 & v24)",
    ("v24", "v60"): "v59 | (v44 & v24)",
    ("v60", "v61"): "v60 & v54",
    ("v54", "v61"): "v60 & v54",
    ("v61", "v62"): "v61 | v47",
    ("v47", "v62"): "v61 | v47",
    ("v62", "v63"): "v62 & v56",
    ("v56", "v63"): "v62 & v56",
    ("v63", "v64"): "v63 | v49",
    ("v49", "v64"): "v63 | v49",
    ("v64", "v65"): "v64 & v58",
    ("v58", "v65"): "v64 & v58",
    ("v65", "v66"): "v65 | (v52 & v28)",
    ("v52", "v66"): "v65 | (v52 & v28)",
    ("v28", "v66"): "v65 | (v52 & v28)",
    ("v66", "v67"): "v66 & v60",
    ("v60", "v67"): "v66 & v60",
    ("v67", "v68"): "v67 | v53",
    ("v53", "v68"): "v67 | v53",
    ("v68", "v69"): "v68 & v62",
    ("v62", "v69"): "v68 & v62",
    ("v69", "v70"): "v69 | v55",
    ("v55", "v70"): "v69 | v55",
    ("v70", "v71"): "v70 & v64",
    ("v64", "v71"): "v70 & v64",
    ("v71", "v72"): "v71 | (v58 & v34)",
    ("v58", "v72"): "v71 | (v58 & v34)",
    ("v34", "v72"): "v71 | (v58 & v34)",
    ("v72", "v73"): "v72 & v66",
    ("v66", "v73"): "v72 & v66",
    ("v73", "v74"): "v73 | v57",
    ("v57", "v74"): "v73 | v57",
    ("v74", "v75"): "v74 & v68",
    ("v68", "v75"): "v74 & v68",
    ("v75", "v76"): "v75 | v61",
    ("v61", "v76"): "v75 | v61",
    ("v76", "v77"): "v76 & v70",
    ("v70", "v77"): "v76 & v70",
    ("v77", "v78"): "v77 | (v63 & v41)",
    ("v63", "v78"): "v77 | (v63 & v41)",
    ("v41", "v78"): "v77 | (v63 & v41)",
    ("v78", "v79"): "v78 & v72",
    ("v72", "v79"): "v78 & v72",
    ("v79", "v80"): "v79 | v65",
    ("v65", "v80"): "v79 | v65",
    ("v80", "v81"): "v80 & v74",
    ("v74", "v81"): "v80 & v74",
    ("v81", "v82"): "v81 | v67",
    ("v67", "v82"): "v81 | v67",
    ("v82", "v83"): "v82 & v76",
    ("v76", "v83"): "v82 & v76",
    ("v83", "v84"): "v83 | (v69 & v45)",
    ("v69", "v84"): "v83 | (v69 & v45)",
    ("v45", "v84"): "v83 | (v69 & v45)",
    ("v84", "v85"): "v84 & v78",
    ("v78", "v85"): "v84 & v78",
    ("v85", "v86"): "v85 | v71",
    ("v71", "v86"): "v85 | v71",
    ("v86", "v87"): "v86 & v80",
    ("v80", "v87"): "v86 & v80",
    ("v87", "v88"): "v87 | v73",
    ("v73", "v88"): "v87 | v73",
    ("v88", "v89"): "v88 & v82",
    ("v82", "v89"): "v88 & v82",
    ("v89", "v90"): "v89 | (v75 & v51)",
    ("v75", "v90"): "v89 | (v75 & v51)",
    ("v51", "v90"): "v89 | (v75 & v51)",
    ("v90", "v91"): "v90 & v84",
    ("v84", "v91"): "v90 & v84",
    ("v91", "v92"): "v91 | v77",
    ("v77", "v92"): "v91 | v77",
    ("v92", "v93"): "v92 & v86",
    ("v86", "v93"): "v92 & v86",
    ("v93", "v94"): "v93 | v79",
    ("v79", "v94"): "v93 | v79",
    ("v94", "v95"): "v94 & v88",
    ("v88", "v95"): "v94 & v88",
    ("v95", "v96"): "v95 | (v81 & v59)",
    ("v81", "v96"): "v95 | (v81 & v59)",
    ("v59", "v96"): "v95 | (v81 & v59)",
    ("v96", "v97"): "v96 & v90",
    ("v90", "v97"): "v96 & v90",
    ("v97", "v98"): "v97 | v83",
    ("v83", "v98"): "v97 | v83",
    ("v98", "v99"): "v98 & v92",
    ("v92", "v99"): "v98 & v92",
    ("v99", "v100"): "v99 | v85",
    ("v85", "v100"): "v99 | v85"
}




def evaluate_state(state, primes, edge_functions):

    #Evaluates the next state of the network according to the primes and edge functions.
    
    new_state = state.copy()

    # Evaluate primes (node functions)
    #print("\nEvaluating node functions:")
    for node, func in primes.items():
        if isinstance(func, str):  # if the function is a string
            try:
                # Do not replace AND and OR for node functions, as they are used for boolean logic
                new_state[node] = eval(func, {},new_state)
                #print(f"Node {node}: {func} => {new_state[node]}")
            except Exception as e:
                print(f"Warning: Error evaluating function for node {node}. Error: {e}")
                new_state[node] = 0  # Default value
        elif callable(func):  # if the function is callable
            new_state[node] = func(state)
            #print(f"Node {node} callable: {new_state[node]}")
        else:  # Here we handle the case where the function is a simple value (like a constant)
            new_state[node] = state[node]  # The state doesn't change for simple nodes
            #print(f"Node {node} simple value: {new_state[node]}")

    # Now, evaluate the edge functions (between nodes)
    #print("\nEvaluating edge functions:")
    for (start_node, end_node), edge_func in edge_functions.items():
        # Check if start_node is in the current state and the end_node is also in the state
        if start_node in state:
            try:
                # Keep the AND and OR as they are (no need to replace)
                new_state[end_node] = eval(edge_func, {}, new_state)
                #print(f"Edge ({start_node} -> {end_node}): {edge_func} => {new_state[end_node]}")
            except Exception as e:
                print(f"Warning: Error evaluating edge function for edge ({start_node}, {end_node}). Error: {e}")
                new_state[end_node] = 0  # Default value

    return new_state



def find_initial_conditions(primes, target_values, nodes, edge_functions, initial_state, max_iterations=100, agent=None):
    """
    Find ONLY the initial conditions (including assignments for the None nodes)
    that lead the network to the target values after updates.

    :param primes: A dictionary of node functions (primes).
    :param target_values: A dictionary of target values for each node.
    :param nodes: A list of nodes in the network.
    :param edge_functions: A dictionary of edge functions.
    :param initial_state: A dictionary with initial values for each node (None => to be randomized).
    :param max_iterations: The maximum number of iterations to simulate (default: 100).
    :param agent: An optional Q-learning agent to guide the search (default: None).
    :return: A list of dictionaries, each containing:
        {
          "initial_condition": dict,        # The assignment (including your fixed values + bits for None)
          "final_state": dict,             # The final stable state after updates
          "intermediate_states": list[dict] # The states after each iteration until stabilization
        }
    """

    # Nodes with a value of None – we will generate all combinations for these nodes
    none_nodes = [node for node in nodes if initial_state.get(node) is None]

    # All possible combinations of 0/1 for the nodes that are None
    possible_replace_none = list(product([0, 1], repeat=len(none_nodes)))

    valid_initial_conditions = []
    visited_states = set()
    print("initial_state:", initial_state)
    print("none_nodes:", none_nodes)
    print("possible_replace_none:", possible_replace_none)

    for condition in possible_replace_none:
        # Build a new initial state for the current combination
        possible_conditions = initial_state.copy()
        for idx, node in enumerate(none_nodes):
            possible_conditions[node] = condition[idx]

        # Ensure we haven't already visited this state
        trial_state_key = tuple(sorted(possible_conditions.items()))
        if trial_state_key in visited_states:
            continue
        visited_states.add(trial_state_key)

        print(f"\n--- Testing initial condition: {possible_conditions} ---")
        current_state = possible_conditions.copy()
        intermediate_states = []

        # Update the network until it stabilizes or the maximum iterations are reached
        for _ in range(max_iterations):
            if agent:
                action = agent.choose_action(current_state)
                next_state = evaluate_state(current_state, primes, edge_functions)
            else:
                next_state = evaluate_state(current_state, primes, edge_functions)

            intermediate_states.append(next_state.copy())

            # If there is no change between the current state and the next state, the network has stabilized
            if next_state == current_state:
                print(f"--> Stabilized final state: {next_state}")
                # Remove the last state from the list since it's identical to the previous one
                intermediate_states.pop()
                break

            current_state = next_state

        # Check if the final state matches the target values
        if all(current_state[node] == val for node, val in target_values.items()):
            # If yes – add the valid initial condition to the list
            valid_initial_conditions.append({
                "initial_condition": possible_conditions.copy(),
                "final_state": current_state.copy(),
                "intermediate_states": intermediate_states
            })

    return valid_initial_conditions


