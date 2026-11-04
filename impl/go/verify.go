package main

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"os"
	"strconv"
	"strings"
	"time"
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

// checkMembers requires all of want, and rejects anything else except x- members.
func checkMembers(m map[string]any, want []string) error {
	for _, w := range want {
		if _, ok := m[w]; !ok {
			return Code("bad_type")
		}
	}
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
	ents, ok1 := m["tensors"].([]any)
	req, ok2 := m["requires"].([]any)
	if !ok1 || !ok2 {
		return Code("bad_type")
	}
	for _, e := range ents {
		em, ok := e.(map[string]any)
		if !ok {
			return Code("bad_type")
		}
		if err := checkMembers(em, tensorMembers); err != nil {
			return err
		}
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
	for i, e := range ents {
		em := e.(map[string]any)
		shape, ok := nonNegInts(em["shape"], -1)
		if !ok {
			return Code("bad_type")
		}
		ed := edit{str(em["name"]), str(em["dtype"]), str(em["base_digest"]), str(em["result_digest"]), shape, nil, nil}
		// val is raw bits; fixtures store it as an unsigned type of equal width.
		idx, val := st.tensors[strconv.Itoa(i)+".idx"], st.tensors[strconv.Itoa(i)+".val"]
		if itemSize[ed.dtype] == 0 || idx.dtype != "I64" || itemSize[val.dtype] != itemSize[ed.dtype] || len(val.raw)/int(itemSize[ed.dtype])*8 != len(idx.raw) {
			return Code("payload_mismatch")
		}
		if digest("payload", idx.raw) != em["idx_digest"] || digest("payload", val.raw) != em["val_digest"] {
			return Code("payload_mismatch")
		}
		ed.val = val.raw
		for j := 0; j < len(idx.raw); j += 8 {
			ed.idx = append(ed.idx, binary.LittleEndian.Uint64(idx.raw[j:]))
		}
		edits[i] = ed
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
	b, err := os.ReadFile(file)
	if err != nil {
		return "", err
	}
	bad := Code("bad_chain")
	if len(b) == 0 || b[len(b)-1] != '\n' {
		return "", bad
	}
	prev := "sha256:" + strings.Repeat("0", 64)
	for seq, line := range bytes.Split(b[:len(b)-1], []byte("\n")) {
		v, err := parseJSON(line)
		e, ok := v.(map[string]any)
		if err != nil || !ok || !bytes.Equal(canon(v), line) || len(e) != 5 {
			return "", bad
		}
		_, isBody := e["body"].(map[string]any)
		_, isKind := e["kind"].(string)
		t, isTime := e["time"].(string)
		if !isBody || !isKind || !isTime || e["seq"] != int64(seq) || e["prev"] != prev {
			return "", bad
		}
		if _, err := time.Parse(time.RFC3339, t); err != nil || !strings.HasSuffix(t, "Z") {
			return "", bad
		}
		prev = digest("ledger", line)
	}
	return prev, nil
}

func verifyReceipt(file string, pub []byte) error {
	raw, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	ev, err := parseJSON(raw)
	env, ok := ev.(map[string]any)
	if err != nil || !ok {
		return Code("bad_json")
	}
	pt, ok1 := env["payloadType"].(string)
	pl, ok2 := env["payload"].(string)
	sigs, ok3 := env["signatures"].([]any)
	if !ok1 || !ok2 || !ok3 {
		return Code("bad_json")
	}
	if pt != "application/vnd.in-toto+json" {
		return Code("bad_type")
	}
	payload, err := base64.StdEncoding.DecodeString(pl)
	if err != nil {
		return Code("bad_json")
	}
	sv, err := parseJSON(payload)
	if err != nil {
		return err
	}
	if !bytes.Equal(canon(sv), payload) {
		return Code("not_canonical")
	}
	s, _ := sv.(map[string]any)
	pred, _ := s["predicate"].(map[string]any)
	if s["_type"] != "https://in-toto.io/Statement/v1" || s["predicateType"] != "https://neyra.run/receipt/v1" {
		return Code("bad_type")
	}
	claim, _ := pred["claim"].(map[string]any)
	subj, _ := s["subject"].([]any)
	if len(subj) != 2 || claim == nil {
		return Code("subject_mismatch")
	}
	for i, want := range [][2]string{{"result-weights", str(claim["result_root"])}, {"patch", str(claim["patch_id"])}} {
		sm, _ := subj[i].(map[string]any)
		d, _ := sm["digest"].(map[string]any)
		if sm["name"] != want[0] || !strings.HasPrefix(want[1], "sha256:") || d["sha256"] != strings.TrimPrefix(want[1], "sha256:") {
			return Code("subject_mismatch")
		}
	}
	if pub == nil {
		return nil
	}
	pae := []byte("DSSEv1 " + strconv.Itoa(len(pt)) + " " + pt + " " + strconv.Itoa(len(payload)) + " ")
	pae = append(pae, payload...)
	keyid := digest("key", pub)
	for _, sg := range sigs {
		m, _ := sg.(map[string]any)
		sig, err := base64.StdEncoding.DecodeString(str(m["sig"]))
		if m["keyid"] == keyid && err == nil && len(pub) == ed25519.PublicKeySize && ed25519.Verify(pub, pae, sig) {
			return nil
		}
	}
	return Code("bad_signature")
}

func unhex(s string) []byte { b, _ := hex.DecodeString(s); return b }
