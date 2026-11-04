package main

import (
	"bytes"
	"sort"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"
)

// Code is a protocol error code; the zero value is not used, success is "ok".
type Code string

func (c Code) Error() string { return string(c) }

const maxInt = 1<<53 - 1

// Parsed JSON values are nil, bool, string, int64, []any and map[string]any.
// Objects lose key order on purpose: canonical form never depends on it.

type parser struct {
	s     []byte
	i     int
	depth int
}

// parseJSON is the strict profile of section 2. Anything it rejects is bad_json.
func parseJSON(b []byte) (any, error) {
	p := &parser{s: b}
	p.ws()
	v, err := p.value()
	if err != nil {
		return nil, err
	}
	p.ws()
	if p.i != len(p.s) {
		return nil, Code("bad_json")
	}
	return v, nil
}

func (p *parser) ws() {
	for p.i < len(p.s) && bytes.IndexByte([]byte(" \t\n\r"), p.s[p.i]) >= 0 {
		p.i++
	}
}

func (p *parser) lit(w string, v any) (any, error) {
	if bytes.HasPrefix(p.s[p.i:], []byte(w)) {
		p.i += len(w)
		return v, nil
	}
	return nil, Code("bad_json")
}

func (p *parser) value() (any, error) {
	if p.i >= len(p.s) {
		return nil, Code("bad_json")
	}
	switch c := p.s[p.i]; {
	case c == '{':
		return p.object()
	case c == '[':
		return p.array()
	case c == '"':
		return p.str()
	case c == 't':
		return p.lit("true", true)
	case c == 'f':
		return p.lit("false", false)
	case c == 'n':
		return p.lit("null", nil)
	case c == '-' || (c >= '0' && c <= '9'):
		return p.number()
	}
	return nil, Code("bad_json")
}

func (p *parser) number() (any, error) {
	start := p.i
	if p.s[p.i] == '-' {
		p.i++
	}
	d := p.i
	for p.i < len(p.s) && p.s[p.i] >= '0' && p.s[p.i] <= '9' {
		p.i++
	}
	n := p.i - d
	if n == 0 || (n > 1 && p.s[d] == '0') || n > 16 {
		return nil, Code("bad_json")
	}
	if p.i < len(p.s) && bytes.IndexByte([]byte(".eE"), p.s[p.i]) >= 0 {
		return nil, Code("bad_json")
	}
	v, _ := strconv.ParseInt(string(p.s[start:p.i]), 10, 64)
	if v > maxInt || v < -maxInt {
		return nil, Code("bad_json")
	}
	return v, nil
}

func (p *parser) hex4() (rune, bool) {
	if p.i+4 > len(p.s) {
		return 0, false
	}
	v, err := strconv.ParseUint(string(p.s[p.i:p.i+4]), 16, 32)
	if err != nil || p.s[p.i] == '+' {
		return 0, false
	}
	p.i += 4
	return rune(v), true
}

func (p *parser) str() (any, error) {
	p.i++ // opening quote
	var out []byte
	for p.i < len(p.s) {
		c := p.s[p.i]
		switch {
		case c == '"':
			p.i++
			return string(out), nil
		case c < 0x20:
			return nil, Code("bad_json")
		case c != '\\':
			r, n := utf8.DecodeRune(p.s[p.i:])
			if r == utf8.RuneError && n == 1 {
				return nil, Code("bad_json")
			}
			out = append(out, p.s[p.i:p.i+n]...)
			p.i += n
			continue
		}
		p.i++
		if p.i >= len(p.s) {
			break
		}
		e := p.s[p.i]
		p.i++
		if r, ok := simpleEscapes[e]; ok {
			out = append(out, r)
			continue
		}
		if e != 'u' {
			return nil, Code("bad_json")
		}
		r, ok := p.hex4()
		if !ok {
			return nil, Code("bad_json")
		}
		if utf16.IsSurrogate(r) {
			// A high surrogate must be followed by an escaped low one.
			if r >= 0xdc00 || !bytes.HasPrefix(p.s[p.i:], []byte(`\u`)) {
				return nil, Code("bad_json")
			}
			p.i += 2
			lo, ok := p.hex4()
			if !ok || lo < 0xdc00 || lo > 0xdfff {
				return nil, Code("bad_json")
			}
			r = utf16.DecodeRune(r, lo)
		}
		out = utf8.AppendRune(out, r)
	}
	return nil, Code("bad_json")
}

var simpleEscapes = map[byte]byte{'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}

func (p *parser) enter() error {
	if p.depth++; p.depth > 128 {
		return Code("bad_json")
	}
	return nil
}

func (p *parser) array() (any, error) {
	if err := p.enter(); err != nil {
		return nil, err
	}
	p.i++
	out := []any{}
	p.ws()
	if p.i < len(p.s) && p.s[p.i] == ']' {
		p.i++
		p.depth--
		return out, nil
	}
	for {
		p.ws()
		v, err := p.value()
		if err != nil {
			return nil, err
		}
		out = append(out, v)
		p.ws()
		if p.i >= len(p.s) {
			return nil, Code("bad_json")
		}
		c := p.s[p.i]
		p.i++
		if c == ']' {
			p.depth--
			return out, nil
		}
		if c != ',' {
			return nil, Code("bad_json")
		}
	}
}

func (p *parser) object() (any, error) {
	if err := p.enter(); err != nil {
		return nil, err
	}
	p.i++
	out := map[string]any{}
	p.ws()
	if p.i < len(p.s) && p.s[p.i] == '}' {
		p.i++
		p.depth--
		return out, nil
	}
	for {
		p.ws()
		if p.i >= len(p.s) || p.s[p.i] != '"' {
			return nil, Code("bad_json")
		}
		k, err := p.str()
		if err != nil {
			return nil, err
		}
		p.ws()
		if p.i >= len(p.s) || p.s[p.i] != ':' {
			return nil, Code("bad_json")
		}
		p.i++
		p.ws()
		v, err := p.value()
		if err != nil {
			return nil, err
		}
		if _, dup := out[k.(string)]; dup {
			return nil, Code("bad_json")
		}
		out[k.(string)] = v
		p.ws()
		if p.i >= len(p.s) {
			return nil, Code("bad_json")
		}
		c := p.s[p.i]
		p.i++
		if c == '}' {
			p.depth--
			return out, nil
		}
		if c != ',' {
			return nil, Code("bad_json")
		}
	}
}

// utf16Less orders strings by UTF-16 code units, which differs from byte order
// for characters above the BMP.
func utf16Less(a, b string) bool {
	x, y := utf16.Encode([]rune(a)), utf16.Encode([]rune(b))
	for i := 0; i < len(x) && i < len(y); i++ {
		if x[i] != y[i] {
			return x[i] < y[i]
		}
	}
	return len(x) < len(y)
}

// canon encodes a parsed value as RFC 8785 JSON.
func canon(v any) []byte { return appendCanon(nil, v) }

func appendCanon(b []byte, v any) []byte {
	switch v := v.(type) {
	case nil:
		return append(b, "null"...)
	case bool:
		return strconv.AppendBool(b, v)
	case int64:
		return strconv.AppendInt(b, v, 10)
	case string:
		return appendStr(b, v)
	case []any:
		b = append(b, '[')
		for i, e := range v {
			if i > 0 {
				b = append(b, ',')
			}
			b = appendCanon(b, e)
		}
		return append(b, ']')
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Slice(keys, func(i, j int) bool { return utf16Less(keys[i], keys[j]) })
		b = append(b, '{')
		for i, k := range keys {
			if i > 0 {
				b = append(b, ',')
			}
			b = appendStr(b, k)
			b = append(b, ':')
			b = appendCanon(b, v[k])
		}
		return append(b, '}')
	}
	panic("canon: unsupported value")
}

func appendStr(b []byte, s string) []byte {
	b = append(b, '"')
	for _, r := range s {
		switch {
		case r == '"' || r == '\\':
			b = append(b, '\\', byte(r))
		case r < 0x20:
			if k := bytes.IndexRune([]byte("\b\t\n\f\r"), r); k >= 0 {
				b = append(b, '\\', "btnfr"[k])
			} else {
				b = append(b, `\u00`...)
				b = append(b, "0123456789abcdef"[r>>4], "0123456789abcdef"[r&15])
			}
		default:
			b = utf8.AppendRune(b, r)
		}
	}
	return append(b, '"')
}
