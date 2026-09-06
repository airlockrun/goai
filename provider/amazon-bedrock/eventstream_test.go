package bedrock

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"hash/crc32"
	"io"
	"testing"
	"testing/iotest"

	goaierrors "github.com/airlockrun/goai/errors"
)

func awsFrame(kind, name, payload string) []byte {
	var h []byte
	key := ":event-type"
	if kind == "exception" {
		key = ":exception-type"
	}
	for _, pair := range [][2]string{{":message-type", kind}, {key, name}, {":content-type", "application/json"}} {
		h = append(h, byte(len(pair[0])))
		h = append(h, pair[0]...)
		h = append(h, 7, byte(len(pair[1])>>8), byte(len(pair[1])))
		h = append(h, pair[1]...)
	}
	b := make([]byte, 12)
	binary.BigEndian.PutUint32(b, uint32(16+len(h)+len(payload)))
	binary.BigEndian.PutUint32(b[4:], uint32(len(h)))
	binary.BigEndian.PutUint32(b[8:], crc32.ChecksumIEEE(b[:8]))
	b = append(b, h...)
	b = append(b, payload...)
	return binary.BigEndian.AppendUint32(b, crc32.ChecksumIEEE(b))
}

func TestReadEventFrame(t *testing.T) {
	frame := awsFrame("event", "contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"Hello"}}`)
	t.Run("fragmented", func(t *testing.T) {
		h, p, err := readEventFrame(iotest.OneByteReader(bytes.NewReader(frame)))
		if err != nil || h[":event-type"] != "contentBlockDelta" || !json.Valid(p) {
			t.Fatalf("%v %s %v", h, p, err)
		}
	})
	t.Run("every truncation", func(t *testing.T) {
		for n := 1; n < len(frame); n++ {
			_, _, err := readEventFrame(bytes.NewReader(frame[:n]))
			if !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("length %d: %v", n, err)
			}
		}
	})
	for _, offset := range []int{0, 8, 12, len(frame) - 1} {
		t.Run("CRC", func(t *testing.T) {
			bad := bytes.Clone(frame)
			bad[offset] ^= 1
			if _, _, err := readEventFrame(bytes.NewReader(bad)); err == nil {
				t.Fatalf("accepted corruption at %d", offset)
			}
		})
	}
	t.Run("invalid length", func(t *testing.T) {
		bad := bytes.Clone(frame)
		binary.BigEndian.PutUint32(bad, 0)
		binary.BigEndian.PutUint32(bad[8:], crc32.ChecksumIEEE(bad[:8]))
		if _, _, err := readEventFrame(bytes.NewReader(bad)); err == nil {
			t.Fatal("accepted zero frame length")
		}
	})
	for _, kind := range []string{"validationException", "throttlingException"} {
		t.Run(kind, func(t *testing.T) {
			_, _, err := readEventFrame(bytes.NewReader(awsFrame("exception", kind, `{"message":"failed"}`)))
			var apiErr *goaierrors.APICallError
			if !errors.As(err, &apiErr) || apiErr.IsRetryable != (kind == "throttlingException") {
				t.Fatalf("%v", err)
			}
		})
	}
}

func TestEventStreamReader(t *testing.T) {
	payload, _ := json.Marshal(map[string][]byte{"bytes": []byte(`{"type":"message_stop"}`)})
	frame := awsFrame("event", "chunk", string(payload))
	r := &eventStreamReader{body: iotest.OneByteReader(bytes.NewReader(append(frame, frame...))), sse: true}
	got, err := io.ReadAll(r)
	if err != nil || string(got) != "data: {\"type\":\"message_stop\"}\n\ndata: {\"type\":\"message_stop\"}\n\n" {
		t.Fatalf("%s %v", got, err)
	}
}
