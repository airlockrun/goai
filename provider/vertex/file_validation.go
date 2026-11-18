package vertex

import (
	"github.com/airlockrun/goai/message"
	"github.com/airlockrun/goai/provider"
	"github.com/airlockrun/goai/provider/google"
)

func (m *VertexLanguageModel) ValidateFile(role message.Role, file message.FilePart) error {
	return google.ValidateFile(m.id, role, file, true)
}

func (m *VertexLanguageModel) ValidateMessages(messages []message.Message) error {
	return provider.ValidateMessageContent(messages, m, true, nil)
}
