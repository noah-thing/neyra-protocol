package main

import (
	"crypto/sha256"
	"encoding/hex"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
)

// Annex C: transformation statements and status records.

const (
	transformType = "https://neyra.run/transformation/v1"
	statusType    = "https://neyra.run/status/v1"
	weightsKey    = "neyra_weights_v1"
)

var (
	kinds         = []string{"patch", "merge", "adapter", "quantize", "edit", "unknown"}
	grades        = []string{"exact", "observed", "sampled", "declared"}
	meanings      = []string{"authored", "reviewed", "approved-for-deploy", "responsible-for-merge"}
	notMeasured   = []string{"training compute", "training data", "behaviour"}
	extentMembers = []string{"tensors_base", "tensors_result", "changed", "added", "removed", "entries_changed", "entries_total"}
	omsIgnore     = []string{".git", ".gitattributes", ".github", ".gitignore"}

	predicateMembers = []string{"type", "kind", "grade", "base", "result", "recipe", "extent", "coverage", "basis",
		"attempts", "rederive", "meaning", "not_measured", "evidence", "time", "tool"}
	statusMembers = []string{"type", "action", "target", "weights_root", "reason", "meaning", "time", "tool"}

	sha256RE = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

func bad(c string) error { return Code(c) }

// exactMembers is true when m has every member of want, and nothing else except x- members if extensible.
func exactMembers(m map[string]any, want []string, extensible bool) bool {
	for _, k := range want {
		if _, ok := m[k]; !ok {
			return false
		}
	}
	for k := range m {
		if !contains(want, k) && !(extensible && strings.HasPrefix(k, "x-")) {
			return false
		}
	}
	return true
}

// basisFor is what a grade fixes: which fields a reader can recompute and which rest on the signer's word.
func basisFor(grade string) map[string]any {
	checkable := []any{"base.weights_root", "result.weights_root", "result.files", "extent"}
	asserted := []any{"base.files", "kind", "recipe", "meaning", "time", "tool", "evidence", "attempts"}
	recomputable := []any{}
	switch grade {
	case "exact":
		recomputable = append(append(recomputable, checkable...), "patch_application")
	case "observed":
		recomputable = append(recomputable, checkable...)
	default:
		asserted = append(asserted, checkable...)
	}
	return map[string]any{"recomputable": recomputable, "asserted": asserted}
}

func isCount(v any) bool {
	n, ok := v.(int64)
	return ok && n >= 0
}

func isTime(v any) bool { s, ok := v.(string); return ok && timeRE.MatchString(s) }

func isTool(v any) bool {
	t, ok := v.(map[string]any)
	if !ok || !exactMembers(t, []string{"name", "version"}, false) {
		return false
	}
	_, a := t["name"].(string)
	_, b := t["version"].(string)
	return a && b
}

// pathLess orders paths component by component, comparing code points, as OMS orders files.
func pathLess(a, b string) bool {
	x, y := strings.Split(a, "/"), strings.Split(b, "/")
	for i := 0; i < len(x) && i < len(y); i++ {
		if x[i] != y[i] {
			return x[i] < y[i]
		}
	}
	return len(x) < len(y)
}

type fileEntry struct{ path, sha string }

// readFiles checks a list of {path, sha256} entries and returns them.
func readFiles(v any) ([]fileEntry, bool) {
	l, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([]fileEntry, 0, len(l))
	for _, e := range l {
		m, ok := e.(map[string]any)
		if !ok || !exactMembers(m, []string{"path", "sha256"}, false) {
			return nil, false
		}
		p, ok1 := m["path"].(string)
		h, ok2 := m["sha256"].(string)
		if !ok1 || !ok2 || !sha256RE.MatchString(h) || strings.Contains(p, "\\") {
			return nil, false
		}
		for _, part := range strings.Split(p, "/") {
			if part == "" || part == "." || part == ".." {
				return nil, false
			}
		}
		out = append(out, fileEntry{p, h})
	}
	for i := 1; i < len(out); i++ {
		if !pathLess(out[i-1].path, out[i].path) {
			return nil, false
		}
	}
	return out, true
}

func omsRoot(files []fileEntry) string {
	h := sha256.New()
	for _, f := range files {
		raw, _ := hex.DecodeString(f.sha)
		h.Write(raw)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func checkTransformPredicate(p map[string]any) error {
	e := bad("bad_transform")
	if p == nil || !exactMembers(p, predicateMembers, true) || p["type"] != "neyra.transformation/1" {
		return e
	}
	kind, grade, meaning := str(p["kind"]), str(p["grade"]), str(p["meaning"])
	if !contains(kinds, kind) || !contains(grades, grade) || !contains(meanings, meaning) {
		return e
	}
	for _, side := range []string{"base", "result"} {
		m, ok := p[side].(map[string]any)
		want := []string{"weights_root", "files"}
		if side == "result" {
			want = append(want, "name", "ignore")
		}
		if !ok || !exactMembers(m, want, false) || !isDigest(m["weights_root"]) {
			return e
		}
		if _, ok := readFiles(m["files"]); !ok {
			return e
		}
	}
	result := p["result"].(map[string]any)
	name, ok := result["name"].(string)
	if !ok || (len(result["files"].([]any)) > 0 && name == "") {
		return e
	}
	ignore, ok := result["ignore"].([]any)
	if !ok {
		return e
	}
	for _, s := range ignore {
		if x, ok := s.(string); !ok || x == "" {
			return e
		}
	}
	recipe, ok := p["recipe"].(map[string]any)
	if !ok {
		return e
	}
	if grade == "exact" && (kind == "patch" || kind == "edit" || kind == "merge") && !isDigest(recipe["patch_id"]) {
		return e
	}
	if p["extent"] == nil {
		if grade != "declared" {
			return e
		}
	} else {
		x, ok := p["extent"].(map[string]any)
		if !ok || !exactMembers(x, extentMembers, false) {
			return e
		}
		for _, v := range x {
			if !isCount(v) {
				return e
			}
		}
	}
	cover, ok := p["coverage"].(map[string]any)
	if !ok || !exactMembers(cover, []string{"bytes_read", "bytes_total"}, false) ||
		!isCount(cover["bytes_read"]) || !isCount(cover["bytes_total"]) {
		return e
	}
	read, total := cover["bytes_read"].(int64), cover["bytes_total"].(int64)
	if (grade == "sampled" && read >= total) || (grade == "declared" && read != 0) ||
		((grade == "exact" || grade == "observed") && read != total) {
		return e
	}
	if !reflect.DeepEqual(p["basis"], basisFor(grade)) {
		return e
	}
	if p["attempts"] != nil {
		if n, ok := p["attempts"].(int64); !ok || n < 1 {
			return e
		}
	}
	steps, ok := p["rederive"].([]any)
	if !ok || len(steps) == 0 {
		return e
	}
	for _, s := range steps {
		if x, ok := s.(string); !ok || x == "" {
			return e
		}
	}
	listed, ok := p["not_measured"].([]any)
	if !ok {
		return e
	}
	said := map[string]bool{}
	for _, s := range listed {
		x, ok := s.(string)
		if !ok {
			return e
		}
		said[x] = true
	}
	for _, s := range notMeasured {
		if !said[s] {
			return e
		}
	}
	ev, ok := p["evidence"].([]any)
	if !ok {
		return e
	}
	for _, x := range ev {
		m, ok := x.(map[string]any)
		if !ok || !exactMembers(m, []string{"role", "digest"}, false) || !isDigest(m["digest"]) {
			return e
		}
		if _, ok := m["role"].(string); !ok {
			return e
		}
	}
	if !isTime(p["time"]) || !isTool(p["tool"]) || (kind == "unknown" && grade == "exact") {
		return e
	}
	return nil
}

// wantedSubjects is what the predicate implies, as canonical strings.
func wantedSubjects(p map[string]any) []string {
	result := p["result"].(map[string]any)
	files, _ := readFiles(result["files"])
	subject := func(name, key, value string) string {
		return string(canon(map[string]any{"name": name, "digest": map[string]any{key: value}}))
	}
	out := []string{subject("result-weights", weightsKey, strings.TrimPrefix(str(result["weights_root"]), "sha256:"))}
	for _, f := range files {
		out = append(out, subject(f.path, "sha256", f.sha))
	}
	if len(files) > 0 {
		out = append(out, subject(str(result["name"]), "sha256", omsRoot(files)))
	}
	sort.Strings(out)
	return out
}

func sameSubjects(got any, want []string) bool {
	l, ok := got.([]any)
	if !ok || len(l) != len(want) {
		return false
	}
	have := make([]string, len(l))
	for i, s := range l {
		have[i] = string(canon(s))
	}
	sort.Strings(have)
	for i := range have {
		if have[i] != want[i] {
			return false
		}
	}
	return true
}

// statement is a transformation statement or status record that passed alone.
type statement struct {
	env  *envelope
	pred map[string]any
}

func readTyped(file string, pub []byte, predicateType string, check func(*envelope, map[string]any) error) (*statement, error) {
	e, err := readEnvelope(file)
	if err != nil {
		return nil, err
	}
	if e.st["_type"] != "https://in-toto.io/Statement/v1" || e.st["predicateType"] != predicateType {
		return nil, Code("bad_type")
	}
	pred, _ := e.st["predicate"].(map[string]any)
	if err := check(e, pred); err != nil {
		return nil, err
	}
	if pub != nil {
		if err := e.verify(pub); err != nil {
			return nil, err
		}
	}
	return &statement{e, pred}, nil
}

func readTransformation(file string, pub []byte) (*statement, error) {
	return readTyped(file, pub, transformType, func(e *envelope, p map[string]any) error {
		if err := checkTransformPredicate(p); err != nil {
			return err
		}
		if !sameSubjects(e.st["subject"], wantedSubjects(p)) {
			return Code("subject_mismatch")
		}
		return nil
	})
}

func readStatus(file string, pub []byte) (*statement, error) {
	return readTyped(file, pub, statusType, func(e *envelope, p map[string]any) error {
		e1 := bad("bad_status")
		action := str(p["action"])
		if p == nil || (action != "supersede" && action != "withdraw") {
			return e1
		}
		want := append([]string{}, statusMembers...)
		if action == "supersede" {
			want = append(want, "replacement")
		}
		if !exactMembers(p, want, true) || p["type"] != "neyra.status/1" {
			return e1
		}
		for _, k := range want {
			if (k == "target" || k == "weights_root" || k == "replacement") && !isDigest(p[k]) {
				return e1
			}
		}
		if r, ok := p["reason"].(string); !ok || r == "" || !contains(meanings, str(p["meaning"])) {
			return e1
		}
		if !isTime(p["time"]) || !isTool(p["tool"]) || (action == "supersede" && p["replacement"] == p["target"]) {
			return e1
		}
		one := string(canon(map[string]any{"name": "result-weights",
			"digest": map[string]any{weightsKey: strings.TrimPrefix(str(p["weights_root"]), "sha256:")}}))
		if !sameSubjects(e.st["subject"], []string{one}) {
			return Code("subject_mismatch")
		}
		return nil
	})
}

func verifyTransform(file string, pub []byte) error {
	_, err := readTransformation(file, pub)
	return err
}

func verifyStatus(file string, pub []byte) error {
	_, err := readStatus(file, pub)
	return err
}

func verifyTransformBinding(statementFile, patchFile string) error {
	s, err := readTransformation(statementFile, nil)
	if err != nil {
		return err
	}
	if err := verifyPatch(patchFile, ""); err != nil {
		return err
	}
	raw, _ := os.ReadFile(patchFile)
	st, _ := readSafetensors(raw)
	manifest := st.meta["neyra.manifest"]
	mv, _ := parseJSON([]byte(manifest))
	m := mv.(map[string]any)
	recipe := s.pred["recipe"].(map[string]any)
	base := s.pred["base"].(map[string]any)
	result := s.pred["result"].(map[string]any)
	if recipe["patch_id"] != digest("patch", []byte(manifest)) || m["base_root"] != base["weights_root"] ||
		m["result_root"] != result["weights_root"] {
		return Code("binding_mismatch")
	}
	return nil
}

// listFiles hashes the regular files of a model directory the way OMS serialises it.
func listFiles(dir string, extra []string) ([]fileEntry, error) {
	if real, err := filepath.EvalSymlinks(dir); err == nil {
		dir = real // the directory itself may be reached through a link; only links inside it are refused
	}
	skipped := [][]string{}
	for _, s := range append(append([]string{}, omsIgnore...), extra...) {
		skipped = append(skipped, strings.Split(s, "/"))
	}
	var out []fileEntry
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(dir, path)
		if rel == "." {
			return nil
		}
		parts := strings.Split(filepath.ToSlash(rel), "/")
		for _, s := range skipped {
			if len(parts) >= len(s) && strings.Join(parts[:len(s)], "/") == strings.Join(s, "/") {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}
		if d.Type()&fs.ModeSymlink != 0 {
			return Code("files_mismatch")
		}
		if d.Type().IsRegular() {
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			sum := sha256.Sum256(b)
			out = append(out, fileEntry{strings.Join(parts, "/"), hex.EncodeToString(sum[:])})
		}
		return nil
	})
	sort.Slice(out, func(i, j int) bool { return pathLess(out[i].path, out[j].path) })
	return out, err
}

func verifyTransformFiles(statementFile, modelDir string) error {
	s, err := readTransformation(statementFile, nil)
	if err != nil {
		return err
	}
	ts, err := loadModel(modelDir)
	if err != nil {
		return err
	}
	result := s.pred["result"].(map[string]any)
	if weightsRoot(ts) != result["weights_root"] {
		return Code("result_mismatch")
	}
	want, _ := readFiles(result["files"])
	if len(want) == 0 {
		return nil
	}
	var extra []string
	for _, x := range result["ignore"].([]any) {
		extra = append(extra, x.(string))
	}
	got, err := listFiles(modelDir, extra)
	if err != nil || len(got) != len(want) {
		return Code("files_mismatch")
	}
	for i := range got {
		if got[i] != want[i] {
			return Code("files_mismatch")
		}
	}
	return nil
}

// readTarget returns the payload and result weights root of a transformation statement or a receipt.
func readTarget(file string) ([]byte, string, error) {
	e, err := readEnvelope(file)
	if err != nil {
		return nil, "", err
	}
	if e.st["predicateType"] == transformType {
		s, err := readTransformation(file, nil)
		if err != nil {
			return nil, "", err
		}
		return s.env.payload, str(s.pred["result"].(map[string]any)["weights_root"]), nil
	}
	r, err := readReceipt(file, nil)
	if err != nil {
		return nil, "", err
	}
	return e.payload, str(r.claim["result_root"]), nil
}

func verifyStatusBinding(statusFile, targetFile string) error {
	s, err := readStatus(statusFile, nil)
	if err != nil {
		return err
	}
	payload, root, err := readTarget(targetFile)
	if err != nil {
		return err
	}
	if s.pred["target"] != digest("statement", payload) || s.pred["weights_root"] != root {
		return Code("binding_mismatch")
	}
	return nil
}
