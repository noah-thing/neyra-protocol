// Command neyra-verify checks Neyra Protocol v1 artifacts.
package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type vector struct {
	Kind, ID  string
	InputHex  string   `json:"input_hex"`
	Domain    string   `json:"domain"`
	DataHex   string   `json:"data_hex"`
	Dtype     string   `json:"dtype"`
	Shape     []uint64 `json:"shape"`
	RawHex    string   `json:"raw_hex"`
	Model     string   `json:"model"`
	File      string   `json:"file"`
	Base      string   `json:"base"`
	PublicHex string   `json:"public_hex"`
	Receipt   string
	Ledger    string
	Patch     string
	Statement string
	Status    string
	Target    string
	LeavesHex []string `json:"leaves_hex"`
	Proof     string
	Data      string
	Expect    struct {
		Code, Hex, Digest, Root, Head string
	}
}

// run returns the observed result in the same shape as vector.Expect.
func run(dir string, v vector) (string, string) {
	path := func(p string) string {
		if p == "" {
			return ""
		}
		return filepath.Join(dir, p)
	}
	switch v.Kind {
	case "jcs":
		pv, err := parseJSON(unhex(v.InputHex))
		if err != nil {
			return string(codeOf(err)), ""
		}
		return "ok", hex.EncodeToString(canon(pv))
	case "digest":
		return "", digest(v.Domain, unhex(v.DataHex))
	case "tensor":
		return "", tensorDigest(tensor{v.Dtype, v.Shape, unhex(v.RawHex)})
	case "root":
		ts, err := loadModel(path(v.Model))
		if err != nil {
			return string(codeOf(err)), ""
		}
		return "", weightsRoot(ts)
	case "patch":
		return string(codeOf(verifyPatch(path(v.File), path(v.Base)))), ""
	case "ledger":
		head, err := verifyLedger(path(v.File))
		return string(codeOf(err)), head
	case "rowtree":
		leaves := make([][]byte, len(v.LeavesHex))
		for i, h := range v.LeavesHex {
			leaves[i] = unhex(h)
		}
		return "", hexDigest(treeRoot(leaves))
	case "rowroot":
		ts, err := loadModel(path(v.Model))
		if err != nil {
			return string(codeOf(err)), ""
		}
		return "", rowsRoot(ts)
	case "rowproof":
		return string(codeOf(verifyRowProof(path(v.Proof), path(v.Data)))), ""
	case "binding":
		return string(codeOf(verifyBinding(path(v.Receipt), path(v.Ledger), path(v.Patch)))), ""
	case "transform", "status":
		var pub []byte
		if v.PublicHex != "" {
			pub = unhex(v.PublicHex)
		}
		if v.Kind == "status" {
			return string(codeOf(verifyStatus(path(v.File), pub))), ""
		}
		return string(codeOf(verifyTransform(path(v.File), pub))), ""
	case "transformbinding":
		return string(codeOf(verifyTransformBinding(path(v.Statement), path(v.Patch)))), ""
	case "transformfiles":
		return string(codeOf(verifyTransformFiles(path(v.Statement), path(v.Model)))), ""
	case "statusbinding":
		return string(codeOf(verifyStatusBinding(path(v.Status), path(v.Target)))), ""
	case "receipt":
		var pub []byte
		if v.PublicHex != "" {
			pub = unhex(v.PublicHex)
		}
		return string(codeOf(verifyReceipt(path(v.File), pub))), ""
	}
	return "unknown kind " + v.Kind, ""
}

func vectors(dir string) int {
	b, err := os.ReadFile(filepath.Join(dir, "cases.json"))
	var cases []vector
	if err == nil {
		err = json.Unmarshal(b, &cases)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	failed := 0
	for _, v := range cases {
		c, val := run(dir, v)
		e := v.Expect
		ok := (e.Code == "" || e.Code == c) && (e.Hex == "" || e.Hex == val) &&
			(e.Digest == "" || e.Digest == val) && (e.Root == "" || e.Root == val) && (e.Head == "" || e.Head == val)
		if e.Code == "ok" && c != "ok" {
			ok = false
		}
		if !ok {
			failed++
			fmt.Printf("FAIL %s/%s: got code=%q value=%q, want %+v\n", v.Kind, v.ID, c, val, e)
		}
	}
	fmt.Printf("%d/%d passed\n", len(cases)-failed, len(cases))
	if failed > 0 {
		return 1
	}
	return 0
}

// guarded reports a panic as a result, so one hostile input cannot hide the rest of a batch.
func guarded(dir string, v vector) (code, value string) {
	defer func() {
		if r := recover(); r != nil {
			code, value = fmt.Sprintf("panic: %v", r), ""
		}
	}()
	return run(dir, v)
}

// results runs every case without comparing, and prints what the verifier observed as one JSON array. The
// differential fuzzer uses it to line this verifier up against another on cases that have no expected value.
func results(dir string) int {
	b, err := os.ReadFile(filepath.Join(dir, "cases.json"))
	var cases []vector
	if err == nil {
		err = json.Unmarshal(b, &cases)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}
	type observed struct {
		Code  string `json:"code"`
		Value string `json:"value"`
	}
	out := make([]observed, len(cases))
	for i, v := range cases {
		out[i].Code, out[i].Value = guarded(dir, v)
	}
	enc, _ := json.Marshal(out)
	fmt.Println(string(enc))
	return 0
}

func main() {
	if len(os.Args) == 3 && os.Args[1] == "vectors" {
		os.Exit(vectors(os.Args[2]))
	}
	if len(os.Args) == 3 && os.Args[1] == "run" {
		os.Exit(results(os.Args[2]))
	}
	fmt.Fprintln(os.Stderr, "usage: neyra-verify vectors|run <path to a folder holding cases.json>")
	os.Exit(2)
}
