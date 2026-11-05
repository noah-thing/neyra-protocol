# neyra-verify

Go verifier for Neyra Protocol v1, standard library only.

    go build -o neyra-verify . && ./neyra-verify vectors ../../spec/vectors

Prints one line per failing case and a total; exit status 0 only if every case passes.
