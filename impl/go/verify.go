package main

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"os"
	"regexp"
	"strconv"
	"strings"
)

var (
	manifestMembers = []string{"type", "base_root", "result_root", "requires", "tensors"}
	tensorMembers   = []string{"name", "dtype", "shape", "base_digest", "result_digest", "idx_digest", "val_digest"}
)

func codeOf(err error) Code {
	if err == nil {
		return "ok"
	}
	if c, ok := err.(Code); ok {
		return c
	}
	return Code(err.Error())
}

// checkMembers rejects anything outside want except x- members.
func checkMembers(m map[string]any, want []string) error {
	for k := range m {
		if !strings.HasPrefix(k, "x-") && !contains(want, k) {
			return Code("unknown_member")
		}
	}
	return nil
}

func contains(l []string, s string) bool {
	for _, x := range l {
		if x == s {
			return true
		}
	}
	return false
}

func str(v any) string { s, _ := v.(string); return s }

var claimMembers = []string{"base_root", "result_root", "patch_id", "task", "eval_env", "statements", "gates", "verdict", "limits"}

var unsignedOf = map[uint64]string{1: "U8", 2: "U16", 4: "U32", 8: "U64"}

var timeRE = regexp.MustCompile(`^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$`)

var digestRE = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

func isDigest(v any) bool { s, ok := v.(string); return ok && digestRE.MatchString(s) }

// manifestShape checks the types of required members once they are known to be present.
func manifestShape(m map[string]any) (ents, req []any, err error) {
	bad := Code("bad_manifest")
	ents, ok1 := m["tensors"].([]any)
	req, ok2 := m["requires"].([]any)
	if !ok1 || !ok2 || !isDigest(m["base_root"]) || !isDigest(m["result_root"]) {
		return nil, nil, bad
	}
	for _, r := range req {
		if _, ok := r.(string); !ok {
			return nil, nil, bad
		}
	}
	for _, e := range ents {
		em, ok := e.(map[string]any)
		if !ok {
			return nil, nil, bad
		}
		_, shapeOK := nonNegInts(em["shape"], -1)
		_, known := itemSize[str(em["dtype"])]
		if _, ok := em["name"].(string); !ok || !shapeOK || !known {
			return nil, nil, bad
		}
		for _, k := range tensorMembers[3:] {
			if !isDigest(em[k]) {
				return nil, nil, bad
			}
		}
	}
	for i := 1; i < len(ents); i++ {
		if !utf16Less(str(ents[i-1].(map[string]any)["name"]), str(ents[i].(map[string]any)["name"])) {
			return nil, nil, bad
		}
	}
	return ents, req, nil
}

func verifyPatch(file, baseDir string) error {
	raw, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	st, err := readSafetensors(raw)
	if err != nil {
		return err
	}
	ms, ok := st.meta["neyra.manifest"]
	if !ok {
		return Code("bad_type")
	}
	mv, err := parseJSON([]byte(ms))
	if err != nil {
		return err
	}
	if !bytes.Equal(canon(mv), []byte(ms)) {
		return Code("not_canonical")
	}
	m, ok := mv.(map[string]any)
	if !ok || m["type"] != "neyra.patch/1" {
		return Code("bad_type")
	}
	if err := checkMembers(m, manifestMembers); err != nil {
		return err
	}
	if es, ok := m["tensors"].([]any); ok {
		for _, e := range es {
			if em, ok := e.(map[string]any); ok {
				if err := checkMembers(em, tensorMembers); err != nil {
					return err
				}
			}
		}
	}
	ents, req, err := manifestShape(m)
	if err != nil {
		return err
	}
	if len(req) > 0 {
		return Code("unknown_extension") // no extension is implemented
	}

	// Payload digests, then index sanity, for every tensor before any base work.
	type edit struct {
		name, dtype, base, result string
		shape                     []uint64
		idx                       []uint64
		val                       []byte
	}
	edits := make([]edit, len(ents))
	digestBad := false // structure of every payload is checked before any digest
	for i, e := range ents {
		em := e.(map[string]any)
		shape, _ := nonNegInts(em["shape"], -1)
		ed := edit{str(em["name"]), str(em["dtype"]), str(em["base_digest"]), str(em["result_digest"]), shape, nil, nil}
		idx, val := st.tensors[strconv.Itoa(i)+".idx"], st.tensors[strconv.Itoa(i)+".val"]
		if idx.dtype != "I64" || val.dtype != unsignedOf[itemSize[ed.dtype]] || len(idx.shape) != 1 ||
			len(val.shape) != 1 || idx.shape[0] != val.shape[0] {
			return Code("bad_manifest")
		}
		if digest("payload", idx.raw) != em["idx_digest"] || digest("payload", val.raw) != em["val_digest"] {
			digestBad = true
		}
		ed.val = val.raw
		for j := 0; j < len(idx.raw); j += 8 {
			ed.idx = append(ed.idx, binary.LittleEndian.Uint64(idx.raw[j:]))
		}
		edits[i] = ed
	}
	if len(st.tensors) != 2*len(ents) {
		return Code("bad_manifest")
	}
	if digestBad {
		return Code("payload_mismatch")
	}
	for _, ed := range edits {
		n, _ := numel(ed.shape)
		for j, v := range ed.idx {
			// Indices are signed on disk, so a negative one shows up as a huge u64.
			if v >= n || (j > 0 && v <= ed.idx[j-1]) {
				return Code("bad_index")
			}
		}
	}
	if baseDir == "" {
		return nil
	}

	base, err := loadModel(baseDir)
	if err != nil {
		return err
	}
	if weightsRoot(base) != m["base_root"] {
		return Code("base_mismatch")
	}
	for _, ed := range edits {
		t, ok := base[ed.name]
		if !ok || t.dtype != ed.dtype || tensorDigest(t) != ed.base {
			return Code("base_mismatch")
		}
		if len(t.shape) != len(ed.shape) {
			return Code("base_mismatch")
		}
		for i := range t.shape {
			if t.shape[i] != ed.shape[i] {
				return Code("base_mismatch")
			}
		}
		w := int(itemSize[t.dtype])
		out := append([]byte(nil), t.raw...)
		for j, v := range ed.idx {
			copy(out[int(v)*w:], ed.val[j*w:(j+1)*w])
		}
		t.raw = out
		if tensorDigest(t) != ed.result {
			return Code("result_mismatch")
		}
		base[ed.name] = t
	}
	if weightsRoot(base) != m["result_root"] {
		return Code("result_mismatch")
	}
	return nil
}

// verifyLedger returns the head digest of a well formed chain.
func verifyLedger(file string) (string, error) {
	head, _, err := readLedger(file)
	return head, err
}

// readLedger also returns each entry digest with its seq.
func readLedger(file string) (string, map[string]int64, error) {
	b, err := os.ReadFile(file)
	if err != nil {
		return "", nil, err
	}
	bad := Code("bad_chain")
	entries := map[string]int64{}
	prev := "sha256:" + strings.Repeat("0", 64)
	if len(b) == 0 {
		return prev, entries, nil
	}
	if b[len(b)-1] != '\n' {
		return "", nil, bad
	}
	for seq, line := range bytes.Split(b[:len(b)-1], []byte("\n")) {
		v, err := parseJSON(line)
		e, ok := v.(map[string]any)
		if err != nil || !ok || !bytes.Equal(canon(v), line) || len(e) != 5 {
			return "", nil, bad
		}
		_, isBody := e["body"].(map[string]any)
		_, isKind := e["kind"].(string)
		t, isTime := e["time"].(string)
		if !isBody || !isKind || !isTime || e["seq"] != int64(seq) || e["prev"] != prev {
			return "", nil, bad
		}
		if !timeRE.MatchString(t) {
			return "", nil, bad
		}
		prev = digest("ledger", line)
		entries[prev] = int64(seq)
	}
	return prev, entries, nil
}

func verifyReceipt(file string, pub []byte) error {
	_, err := readReceipt(file, pub)
	return err
}

// readReceipt returns the claim and predicate of a receipt that passed.
func readReceipt(file string, pub []byte) (*receipt, error) {
	raw, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	ev, err := parseJSON(raw)
	env, ok := ev.(map[string]any)
	if err != nil || !ok {
		return nil, Code("bad_json")
	}
	pt, ok1 := env["payloadType"].(string)
	pl, ok2 := env["payload"].(string)
	sigs, ok3 := env["signatures"].([]any)
	if !ok1 || !ok2 || !ok3 {
		return nil, Code("bad_json")
	}
	if pt != "application/vnd.in-toto+json" {
		return nil, Code("bad_type")
	}
	payload, err := strictB64(pl)
	if err != nil {
		return nil, Code("bad_json")
	}
	sv, err := parseJSON(payload)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(canon(sv), payload) {
		return nil, Code("not_canonical")
	}
	s, _ := sv.(map[string]any)
	pred, _ := s["predicate"].(map[string]any)
	if s["_type"] != "https://in-toto.io/Statement/v1" || s["predicateType"] != "https://neyra.run/receipt/v1" {
		return nil, Code("bad_type")
	}
	claim, _ := pred["claim"].(map[string]any)
	if claim == nil || claim["type"] != "neyra.claim/1" {
		return nil, Code("bad_claim")
	}
	for _, k := range claimMembers {
		if _, ok := claim[k]; !ok {
			return nil, Code("bad_claim")
		}
	}
	for _, k := range []string{"ledger_head", "ledger_len"} {
		if _, ok := pred[k]; !ok {
			return nil, Code("bad_claim")
		}
	}
	if pred["digest_algorithm"] != "neyra/weights/v1" {
		return nil, Code("bad_claim")
	}
	// The spec lists the subjects in a fixed order but accepts either.
	want := map[string]string{"result-weights": str(claim["result_root"]), "patch": str(claim["patch_id"])}
	subj, _ := s["subject"].([]any)
	seen := map[string]bool{}
	for _, e := range subj {
		sm, _ := e.(map[string]any)
		d, _ := sm["digest"].(map[string]any)
		name := str(sm["name"])
		root, known := want[name]
		if !known || seen[name] || !isDigest(root) || d["sha256"] != strings.TrimPrefix(root, "sha256:") {
			return nil, Code("subject_mismatch")
		}
		seen[name] = true
	}
	if len(seen) != 2 {
		return nil, Code("subject_mismatch")
	}
	if pub == nil {
		return &receipt{claim, pred}, nil
	}
	pae := []byte("DSSEv1 " + strconv.Itoa(len(pt)) + " " + pt + " " + strconv.Itoa(len(payload)) + " ")
	pae = append(pae, payload...)
	keyid := digest("key", pub)
	for _, sg := range sigs {
		m, _ := sg.(map[string]any)
		sig, err := strictB64(str(m["sig"]))
		if m["keyid"] == keyid && err == nil && len(pub) == ed25519.PublicKeySize && ed25519.Verify(pub, pae, sig) {
			return &receipt{claim, pred}, nil
		}
	}
	return nil, Code("bad_signature")
}

func unhex(s string) []byte { b, _ := hex.DecodeString(s); return b }

// strictB64 is padded standard base64 with no whitespace; Go's decoder skips CR and LF.
func strictB64(s string) ([]byte, error) {
	if strings.ContainsAny(s, "\r\n") {
		return nil, Code("bad_json")
	}
	return base64.StdEncoding.Strict().DecodeString(s)
}

type receipt struct{ claim, pred map[string]any }

// verifyBinding checks a receipt against its ledger and patch; each must verify alone first.
func verifyBinding(receiptFile, ledgerFile, patchFile string) error {
	r, err := readReceipt(receiptFile, nil)
	if err != nil {
		return err
	}
	_, entries, err := readLedger(ledgerFile)
	if err != nil {
		return err
	}
	if err := verifyPatch(patchFile, ""); err != nil {
		return err
	}
	raw, _ := os.ReadFile(patchFile)
	st, _ := readSafetensors(raw)
	mv, _ := parseJSON([]byte(st.meta["neyra.manifest"]))
	result := mv.(map[string]any)["result_root"]
	bad := Code("binding_mismatch")
	if digest("patch", []byte(st.meta["neyra.manifest"])) != r.claim["patch_id"] || result != r.claim["result_root"] {
		return bad
	}
	seq, ok := entries[str(r.pred["ledger_head"])]
	if !ok || r.pred["ledger_len"] != seq+1 {
		return bad
	}
	return nil
}
