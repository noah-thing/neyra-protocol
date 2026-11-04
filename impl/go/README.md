# neyra-verify

Independent Go verifier for Neyra Protocol v1, standard library only, written from `spec/PROTOCOL.md` alone.

    go build -o neyra-verify . && ./neyra-verify vectors ../spec/vectors

Prints one line per failing case and a total; exit status 0 only if every case passes.
Open questions the spec left to the implementer are recorded in `../FINDINGS.md`.
