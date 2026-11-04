package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math/bits"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var itemSize = map[string]uint64{
	"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
	"U16": 2, "I16": 2, "F16": 2, "BF16": 2,
	"U32": 4, "I32": 4, "F32": 4,
	"U64": 8, "I64": 8, "F64": 8,
}

type tensor struct {
	dtype string
	shape []uint64
	raw   []byte
}

type stFile struct {
	meta    map[string]string
	hasMeta bool
	tensors map[string]tensor
}

func (t tensor) count() (n uint64, ok bool) { return numel(t.shape) }

func numel(shape []uint64) (uint64, bool) {
	n := uint64(1)
	for _, d := range shape {
		hi, lo := bits.Mul64(n, d)
		if hi != 0 {
			return 0, false
		}
		n = lo
	}
	return n, true
}

func digest(domain string, data []byte) string {
	h := sha256.New()
	h.Write([]byte("neyra/" + domain + "/v1\x00"))
	h.Write(data)
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func tensorDigest(t tensor) string {
	h := sha256.New()
	h.Write([]byte("neyra/tensor/v1\x00" + t.dtype + "\x00"))
	h.Write([]byte{byte(len(t.shape))})
	for _, d := range t.shape {
		h.Write(binary.LittleEndian.AppendUint64(nil, d))
	}
	h.Write(t.raw)
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

func nonNegInts(v any, n int) ([]uint64, bool) {
	a, ok := v.([]any)
	if !ok || (n >= 0 && len(a) != n) {
		return nil, false
	}
	out := make([]uint64, len(a))
	for i, e := range a {
		x, ok := e.(int64)
		if !ok || x < 0 {
			return nil, false
		}
		out[i] = uint64(x)
	}
	return out, true
}

// readSafetensors applies every rejection rule of section 4.
func readSafetensors(b []byte) (*stFile, error) {
	bad := Code("bad_safetensors")
	if len(b) < 8 {
		return nil, bad
	}
	n := binary.LittleEndian.Uint64(b)
	if n > 100_000_000 || n > uint64(len(b)-8) {
		return nil, bad
	}
	hv, err := parseJSON(b[8 : 8+n])
	hdr, ok := hv.(map[string]any)
	if err != nil || !ok {
		return nil, bad
	}
	data := b[8+n:]
	f := &stFile{tensors: map[string]tensor{}}
	type span struct{ start, end uint64 }
	var spans []span
	for name, v := range hdr {
		if name == "__metadata__" {
			m, ok := v.(map[string]any)
			if !ok {
				return nil, bad
			}
			f.hasMeta, f.meta = true, map[string]string{}
			for k, s := range m {
				str, ok := s.(string)
				if !ok {
					return nil, bad
				}
				f.meta[k] = str
			}
			continue
		}
		e, ok := v.(map[string]any)
		if !ok {
			return nil, bad
		}
		dtype, _ := e["dtype"].(string)
		size, known := itemSize[dtype]
		shape, ok1 := nonNegInts(e["shape"], -1)
		off, ok2 := nonNegInts(e["data_offsets"], 2)
		cnt, ok3 := numel(shape)
		if !known || !ok1 || !ok2 || !ok3 || off[1] < off[0] || off[1]-off[0] != cnt*size || cnt > 1<<60/size {
			return nil, bad
		}
		spans = append(spans, span{off[0], off[1]})
		f.tensors[name] = tensor{dtype, shape, nil}
		hdr[name] = off // reuse the map to carry offsets to the second pass
	}
	sort.Slice(spans, func(i, j int) bool {
		return spans[i].start < spans[j].start || (spans[i].start == spans[j].start && spans[i].end < spans[j].end)
	})
	pos := uint64(0)
	for _, s := range spans {
		if s.start != pos {
			return nil, bad
		}
		pos = s.end
	}
	if pos != uint64(len(data)) {
		return nil, bad
	}
	for name, t := range f.tensors {
		off := hdr[name].([]uint64)
		t.raw = data[off[0]:off[1]]
		f.tensors[name] = t
	}
	return f, nil
}

// loadModel merges every shard in dir; sharding does not affect the result.
func loadModel(dir string) (map[string]tensor, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	all := map[string]tensor{}
	files := 0
	for _, de := range ents {
		n := de.Name()
		if !de.Type().IsRegular() || strings.HasPrefix(n, ".") || !strings.HasSuffix(n, ".safetensors") {
			continue
		}
		files++
		b, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			return nil, err
		}
		f, err := readSafetensors(b)
		if err != nil {
			return nil, err
		}
		for k, t := range f.tensors {
			if _, dup := all[k]; dup {
				return nil, Code("bad_safetensors")
			}
			all[k] = t
		}
	}
	if files == 0 {
		return nil, Code("bad_safetensors")
	}
	return all, nil
}

func weightsRoot(ts map[string]tensor) string {
	names := make([]string, 0, len(ts))
	for n := range ts {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool { return utf16Less(names[i], names[j]) })
	arr := make([]any, 0, len(names))
	for _, n := range names {
		t := ts[n]
		shape := make([]any, len(t.shape))
		for i, d := range t.shape {
			shape[i] = int64(d)
		}
		arr = append(arr, map[string]any{"name": n, "dtype": t.dtype, "shape": shape, "digest": tensorDigest(t)})
	}
	return digest("weights", canon(arr))
}
