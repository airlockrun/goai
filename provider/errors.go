package provider

import "errors"

// ErrResponseFormatUnsupported is returned by a provider that cannot honor a
// requested stream.ResponseFormat. Wrap with fmt.Errorf("%w: details", ...) to
// add provider context. Callers match with errors.Is.
var ErrResponseFormatUnsupported = errors.New("provider does not support requested ResponseFormat")
