def process_keyval_args(glvars, argv):
    """
    python XXX.py key1=val1 key2=val2

    puts python variable called "key1" with appropriate value "val1" in the global scope, as if it were defined in the python program file itself
    """
    if len(argv) > 0:
        for ii in range(len(argv)):
            print(argv[ii])
            kyval=argv[ii].split("=")

            if len(kyval) != 2:
                print("error")
            ky  = kyval[0]
            val = kyval[1]

            if type(glvars[ky]) == int:
                glvars[ky] = int(val)
            elif type(glvars[ky]) == str:
                glvars[ky] = val
            elif type(glvars[ky]) == bool:
                glvars[ky] = True if (val == "True") else False
