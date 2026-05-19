package tool

import (
	"errors"
	"strings"

	"github.com/airlockrun/goai/message"
)

// DeniedError marks a tool call the user or policy refused to run. A tool
// Execute returning a DeniedError (directly or wrapped) is classified as
// message.ExecutionDeniedOutput rather than an error — kept distinct so the
// model re-reasons correctly and the UI can show it differently.
type DeniedError struct {
	Reason string
}

func (e DeniedError) Error() string {
	if e.Reason != "" {
		return e.Reason
	}
	return "tool execution denied"
}

// OutputForError classifies a non-nil tool error into the matching
// ToolResultOutput variant: a DeniedError (directly or wrapped) →
// ExecutionDeniedOutput; anything else → ErrorTextOutput. This is the single
// error-classification rule used everywhere a tool error is turned into a
// result.
func OutputForError(err error) message.ToolResultOutput {
	var d DeniedError
	if errors.As(err, &d) {
		return message.ExecutionDeniedOutput{Reason: d.Reason}
	}
	return message.ErrorTextOutput{Value: err.Error()}
}

// SuccessOutput builds the success ToolResultOutput for a Result:
// a ContentOutput when there are attachments (a leading text item plus
// file/image-data items), otherwise a plain TextOutput.
func SuccessOutput(r Result) message.ToolResultOutput {
	if len(r.Attachments) == 0 {
		return message.TextOutput{Value: r.Output}
	}
	items := make([]message.ToolContentItem, 0, len(r.Attachments)+1)
	if r.Output != "" {
		items = append(items, message.ToolContentItem{Type: "text", Text: r.Output})
	}
	for _, a := range r.Attachments {
		if strings.HasPrefix(a.MimeType, "image/") {
			items = append(items, message.ToolContentItem{
				Type:      "image-data",
				Data:      a.Data,
				MediaType: a.MimeType,
			})
		} else {
			items = append(items, message.ToolContentItem{
				Type:      "file-data",
				Data:      a.Data,
				MediaType: a.MimeType,
				Filename:  a.Filename,
			})
		}
	}
	return message.ContentOutput{Value: items}
}
