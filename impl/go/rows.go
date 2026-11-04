package main

import (
	"crypto/sha256"
	"encoding/hex"
	"math/bits"
	"os"
	"sort"
)

const leafBytes = 4096

func rowHash(domain string, data ...[]byte) []byte {
	h := sha256.New()
	h.Write([]byte("neyra/" + domain + "/v1\x00"))
	for _, d := range data {
		h.Write(d)
	}
	return h.Sum(nil)
}

func hexDigest(h []byte) string { return "sha256:" + hex.EncodeToString(h) }

// splitPoint is the largest power of two below n, for n of at least 2.
func splitPoint(n int) int { return 1 << (bits.Len(uint(n-1)) - 1) }

func treeRoot(leaves [][]byte) []byte {
	switch len(leaves) {
	case 0:
		return rowHash("rownode")
	case 1:
		return leaves[0]
	}
	k := splitPoint(len(leaves))
	return rowHash("rownode", treeRoot(leaves[:k]), treeRoot(leaves[k:]))
}

type geometry struct{ rows, width, group uint64 }

func geometryOf(dtype string, shape []uint64) geometry {
	g := geometry{rows: 1, width: itemSize[dtype]}
	if len(shape) > 0 {
		g.rows = shape[0]
		for _, d := range shape[1:] {
			g.width *= d
		}
	}
	if g.rows*g.width != 0 {
		g.group = (leafBytes + g.width - 1) / g.width
	}
	return g
}

func (g geometry) leaves() int {
	if g.group == 0 {
		return 0
	}
	return int((g.rows + g.group - 1) / g.group)
}

func (g geometry) span(k int) (uint64, uint64) {
	end := (uint64(k) + 1) * g.group
	if end > g.rows {
		end = g.rows
	}
	return uint64(k) * g.group * g.width, end * g.width
}

func tensorLeaf(name, dtype string, shape []uint64, tree []byte) []byte {
	dims := make([]any, len(shape))
	for i, d := range shape {
		dims[i] = int64(d)
	}
	obj := map[string]any{"name": name, "dtype": dtype, "shape": dims, "tree": hexDigest(tree)}
	return rowHash("rowtensor", canon(obj))
}

func tensorTree(t tensor) []byte {
	g := geometryOf(t.dtype, t.shape)
	hashes := make([][]byte, g.leaves())
	for k := range hashes {
		a, b := g.span(k)
		hashes[k] = rowHash("rowleaf", t.raw[a:b])
	}
	return treeRoot(hashes)
}

func rowsRoot(ts map[string]tensor) string {
	names := make([]string, 0, len(ts))
	for n := range ts {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool { return utf16Less(names[i], names[j]) })
	leaves := make([][]byte, len(names))
	for i, n := range names {
		t := ts[n]
		leaves[i] = tensorLeaf(n, t.dtype, t.shape, tensorTree(t))
	}
	return hexDigest(treeRoot(leaves))
}

// closeLeaves rebuilds a root of n leaves from known ones and the spare hashes, consumed left to right.
func closeLeaves(n int, known map[int][]byte, spare [][]byte) ([]byte, error) {
	bad := Code("bad_proof")
	var walk func(lo, hi int) ([]byte, error)
	walk = func(lo, hi int) ([]byte, error) {
		has := false
		for k := range known {
			if k >= lo && k < hi {
				has = true
			}
		}
		if !has {
			if len(spare) == 0 {
				return nil, bad
			}
			h := spare[0]
			spare = spare[1:]
			return h, nil
		}
		if hi-lo == 1 {
			return known[lo], nil
		}
		mid := lo + splitPoint(hi-lo)
		l, err := walk(lo, mid)
		if err != nil {
			return nil, err
		}
		r, err := walk(mid, hi)
		if err != nil {
			return nil, err
		}
		return rowHash("rownode", l, r), nil
	}
	root, err := walk(0, n)
	if err != nil {
		return nil, err
	}
	if len(spare) != 0 {
		return nil, bad
	}
	return root, nil
}

func digestList(v any) ([][]byte, bool) {
	a, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([][]byte, len(a))
	for i, e := range a {
		if !isDigest(e) {
			return nil, false
		}
		out[i], _ = hex.DecodeString(e.(string)[7:])
	}
	return out, true
}

func verifyRowProof(proofFile, dataFile string) error {
	bad := Code("bad_proof")
	raw, err := os.ReadFile(proofFile)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(dataFile)
	if err != nil {
		return err
	}
	v, err := parseJSON(raw)
	if err != nil {
		return bad
	}
	p, ok := v.(map[string]any)
	if !ok || len(p) != 8 || p["type"] != "neyra.rowproof/1" {
		return bad
	}
	t, ok := p["tensor"].(map[string]any)
	name, ok1 := t["name"].(string)
	dtype, _ := t["dtype"].(string)
	shape, ok2 := nonNegInts(t["shape"], -1)
	_, known := itemSize[dtype]
	if !ok || len(t) != 3 || !ok1 || !ok2 || !known {
		return bad
	}
	index, ok1 := p["index"].(int64)
	count, ok2 := p["count"].(int64)
	if !ok1 || !ok2 || index < 0 || index >= count {
		return bad
	}
	rs, ok := p["ranges"].([]any)
	if !ok || len(rs) == 0 || !isDigest(p["rows_root"]) {
		return bad
	}
	tree, ok1 := digestList(p["tree"])
	path, ok2 := digestList(p["path"])
	if !ok1 || !ok2 {
		return bad
	}
	g := geometryOf(dtype, shape)
	if g.group == 0 {
		return bad
	}
	var ranges [][2]uint64
	cursor := uint64(0)
	for _, r := range rs {
		ends, ok := nonNegInts(r, 2)
		if !ok || ends[0] < cursor || ends[0] >= ends[1] || ends[1] > g.rows {
			return bad
		}
		cursor = ends[1]
		ranges = append(ranges, [2]uint64{ends[0], ends[1]})
	}
	seen := map[int]bool{}
	var leaves []int
	for _, r := range ranges {
		for k := int(r[0] / g.group); k <= int((r[1]-1)/g.group); k++ {
			if !seen[k] {
				seen[k] = true
				leaves = append(leaves, k)
			}
		}
	}
	sort.Ints(leaves)
	total := uint64(0)
	for _, k := range leaves {
		a, b := g.span(k)
		total += b - a
	}
	if uint64(len(data)) != total {
		return bad
	}
	hashes := map[int][]byte{}
	at := uint64(0)
	for _, k := range leaves {
		a, b := g.span(k)
		hashes[k] = rowHash("rowleaf", data[at:at+b-a])
		at += b - a
	}
	top, err := closeLeaves(g.leaves(), hashes, tree)
	if err != nil {
		return err
	}
	root, err := closeLeaves(int(count), map[int][]byte{int(index): tensorLeaf(name, dtype, shape, top)}, path)
	if err != nil {
		return err
	}
	if hexDigest(root) != p["rows_root"] {
		return Code("proof_mismatch")
	}
	return nil
}
