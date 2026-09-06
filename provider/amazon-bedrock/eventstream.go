package bedrock

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"

	goaierrors "github.com/airlockrun/goai/errors"
)

// eventStreamReader validates AWS frames before exposing InvokeModel payloads.
type eventStreamReader struct {
	body    io.Reader
	pending []byte
	sse     bool
}

func readEventFrame(r io.Reader) (map[string]string, []byte, error) {
	prelude := make([]byte, 12)
	if _, err := io.ReadFull(r, prelude); err != nil {
		return nil, nil, err
	}
	total, headerLen := binary.BigEndian.Uint32(prelude), binary.BigEndian.Uint32(prelude[4:])
	if crc32.ChecksumIEEE(prelude[:8]) != binary.BigEndian.Uint32(prelude[8:]) {
		return nil, nil, errors.New("AWS EventStream prelude CRC mismatch")
	}
	if total < 16 || total > 24*1024*1024 || headerLen > total-16 {
		return nil, nil, errors.New("invalid AWS EventStream frame length")
	}
	frame := make([]byte, int(total))
	copy(frame, prelude)
	if _, err := io.ReadFull(r, frame[12:]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, nil, err
	}
	if crc32.ChecksumIEEE(frame[:total-4]) != binary.BigEndian.Uint32(frame[total-4:]) {
		return nil, nil, errors.New("AWS EventStream message CRC mismatch")
	}
	headers := map[string]string{}
	for h := frame[12 : 12+headerLen]; len(h) > 0; {
		n := int(h[0])
		h = h[1:]
		if n == 0 || len(h) < n+1 {
			return nil, nil, errors.New("invalid AWS EventStream header")
		}
		name, kind := string(h[:n]), h[n]
		h = h[n+1:]
		size := 0
		switch kind {
		case 0, 1:
		case 2:
			size = 1
		case 3:
			size = 2
		case 4:
			size = 4
		case 5, 8:
			size = 8
		case 9:
			size = 16
		case 6, 7:
			if len(h) < 2 {
				return nil, nil, io.ErrUnexpectedEOF
			}
			size = int(binary.BigEndian.Uint16(h))
			h = h[2:]
		default:
			return nil, nil, errors.New("invalid AWS EventStream header type")
		}
		if len(h) < size {
			return nil, nil, io.ErrUnexpectedEOF
		}
		if kind == 7 {
			headers[name] = string(h[:size])
		}
		h = h[size:]
	}
	payload := frame[12+headerLen : total-4]
	if headers[":message-type"] == "exception" || headers[":message-type"] == "error" {
		kind := headers[":exception-type"]
		if kind == "" {
			kind = headers[":error-code"]
		}
		return nil, nil, goaierrors.NewAPICallError(goaierrors.APICallErrorOptions{Message: fmt.Sprintf("Bedrock %s: %s", kind, payload), ResponseBody: string(payload), IsRetryable: kind == "throttlingException" || kind == "internalServerException" || kind == "modelStreamErrorException", IsRetryableSet: true})
	}
	if headers[":message-type"] != "event" || headers[":event-type"] == "" {
		return nil, nil, errors.New("invalid AWS EventStream event headers")
	}
	return headers, payload, nil
}

func (r *eventStreamReader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	for len(r.pending) == 0 {
		headers, payload, err := readEventFrame(r.body)
		if err != nil {
			return 0, err
		}
		if headers[":event-type"] != "chunk" {
			return 0, fmt.Errorf("unexpected Bedrock event %q", headers[":event-type"])
		}
		var chunk struct {
			Bytes []byte `json:"bytes"`
		}
		if err := json.Unmarshal(payload, &chunk); err != nil {
			return 0, err
		}
		if !json.Valid(chunk.Bytes) {
			return 0, errors.New("invalid Bedrock chunk JSON")
		}
		if r.sse {
			r.pending = append([]byte("data: "), chunk.Bytes...)
			r.pending = append(r.pending, '\n', '\n')
		} else {
			r.pending = append(chunk.Bytes, '\n')
		}
	}
	n := copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}
