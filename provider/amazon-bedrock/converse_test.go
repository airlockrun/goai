package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/stream"
)

func TestBedrockRoutingBinaryStream(t *testing.T) {
	original := http.DefaultClient
	t.Cleanup(func() { http.DefaultClient = original })
	for _, id := range []string{"us.anthropic.claude-sonnet-5-v1:0", "arn:aws:bedrock:us-east-1:123:inference-profile/eu.anthropic.claude-opus-5-v1:0", "amazon.nova-pro-v1:0", "global.amazon.nova-2-lite-v1:0", "arn:aws:bedrock:us-east-1:123:application-inference-profile/abc"} {
		t.Run(id, func(t *testing.T) {
			claude := strings.Contains(id, "anthropic.")
			http.DefaultClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				endpoint := "converse-stream"
				if claude {
					endpoint = "invoke-with-response-stream"
				}
				if req.URL.EscapedPath() != "/model/"+strings.ReplaceAll(url.QueryEscape(id), "+", "%20")+"/"+endpoint {
					t.Fatalf("path %s", req.URL.EscapedPath())
				}
				var body map[string]any
				if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if claude {
					if body["anthropic_version"] != "bedrock-2023-05-31" {
						t.Fatalf("%v", body)
					}
				} else if body["anthropic_version"] != nil || body["messages"] == nil {
					t.Fatalf("%v", body)
				}
				var wire []byte
				if claude {
					for _, p := range []string{`{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":3}}}`, `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`, `{"type":"content_block_stop","index":0}`, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`, `{"type":"message_stop"}`} {
						chunk, _ := json.Marshal(map[string][]byte{"bytes": []byte(p)})
						wire = append(wire, awsFrame("event", "chunk", string(chunk))...)
					}
				} else {
					for _, p := range [][2]string{{"messageStart", `{"role":"assistant"}`}, {"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"hello"}}`}, {"contentBlockStop", `{"contentBlockIndex":0}`}, {"messageStop", `{"stopReason":"end_turn"}`}, {"metadata", `{"usage":{"inputTokens":3,"outputTokens":1}}`}} {
						wire = append(wire, awsFrame("event", p[0], p[1])...)
					}
				}
				return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/vnd.amazon.eventstream"}}, Body: io.NopCloser(iotest.OneByteReader(bytes.NewReader(wire)))}, nil
			})}
			events, err := New(Options{AccessKeyID: "key", SecretAccessKey: "secret"}).Model(id).Stream(context.Background(), &stream.CallOptions{Messages: []message.Message{message.NewUserMessage("hi")}})
			if err != nil {
				t.Fatal(err)
			}
			var text string
			finished := false
			for e := range events {
				switch d := e.Data.(type) {
				case stream.ErrorEvent:
					t.Fatal(d.Error)
				case stream.TextDeltaEvent:
					text += d.Text
				case stream.FinishEvent:
					finished = true
					if d.Usage.InputTokens.Total == nil || *d.Usage.InputTokens.Total != 3 {
						t.Fatalf("%+v", d.Usage)
					}
				}
			}
			if text != "hello" || !finished {
				t.Fatalf("text %q finished %v", text, finished)
			}
		})
	}
}

func TestConverseIncompleteStream(t *testing.T) {
	for _, wire := range [][]byte{nil, awsFrame("event", "messageStart", `{"role":"assistant"}`), append(awsFrame("event", "messageStart", `{}`), awsFrame("exception", "validationException", `{"message":"invalid"}`)...)} {
		events := make(chan stream.Event, 20)
		(&BedrockLanguageModel{}).processConverseStream(context.Background(), bytes.NewReader(wire), events, false)
		close(events)
		sawError := false
		for e := range events {
			if e.Type == stream.EventError {
				sawError = true
			}
			if e.Type == stream.EventFinish {
				t.Fatal("false finish")
			}
		}
		if !sawError {
			t.Fatal("missing error")
		}
	}
}
