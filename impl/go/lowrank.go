package main

import (
	"encoding/binary"
	"math"
)

// Annex A arithmetic. Products are rounded to float64 on their own before the add, because Go may fuse x*y+z
// into one rounding unless an explicit conversion separates them, and the annex forbids fused multiply add.

func float64s(raw []byte) []float64 {
	out := make([]float64, len(raw)/8)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[8*i:]))
	}
	return out
}

func halfToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp, mant := uint32(h>>10)&0x1f, uint32(h)&0x3ff
	switch {
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	case exp == 0:
		f := float32(mant) * (1.0 / (1 << 24))
		if sign != 0 {
			return -f
		}
		return f
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}

func float32ToHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp, mant := int(b>>23)&0xff, b&0x7fffff
	if exp == 0xff {
		if mant != 0 {
			return sign | 0x7e00 | uint16(mant>>13)
		}
		return sign | 0x7c00
	}
	e := exp - 127
	switch {
	case e > 15:
		return sign | 0x7c00
	case e >= -14:
		h := uint32(e+15)<<10 | mant>>13
		rem := mant & 0x1fff
		if rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
			h++ // a carry out of the mantissa raises the exponent, and may reach infinity, which is right
		}
		return sign | uint16(h)
	case e < -25:
		return sign
	}
	m := mant | 0x800000
	shift := uint(-e - 1)
	k, rem, half := m>>shift, m&(1<<shift-1), uint32(1)<<(shift-1)
	if rem > half || (rem == half && k&1 == 1) {
		k++
	}
	return sign | uint16(k)
}

func float32ToBF16(f float32) uint16 {
	u := math.Float32bits(f)
	if u&0x7fffffff > 0x7f800000 {
		return uint16(u>>16) | 0x40
	}
	return uint16((u + 0x7fff + (u>>16)&1) >> 16)
}

// applyLowRank returns the tensor bytes after adding the outer products, before idx and val are substituted.
func applyLowRank(dtype string, raw []byte, left, right []float64, rows, cols int) []byte {
	r := len(left) / rows
	w := make([]float64, rows*cols)
	for i := range w {
		switch dtype {
		case "BF16":
			w[i] = float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*i:])) << 16))
		case "F16":
			w[i] = float64(halfToFloat32(binary.LittleEndian.Uint16(raw[2*i:])))
		default:
			w[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
		}
	}
	for k := 0; k < r; k++ {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				w[i*cols+j] += float64(left[i*r+k] * right[k*cols+j])
			}
		}
	}
	out := make([]byte, len(raw))
	for i, x := range w {
		f := float32(x)
		switch dtype {
		case "BF16":
			binary.LittleEndian.PutUint16(out[2*i:], float32ToBF16(f))
		case "F16":
			binary.LittleEndian.PutUint16(out[2*i:], float32ToHalf(f))
		default:
			binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(f))
		}
	}
	return out
}
