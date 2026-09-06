package openaicompat

import "maps"

// FileHeaders returns a copy of configured headers for derived Files clients.
func (p *Provider) FileHeaders() map[string]string { return maps.Clone(p.opts.Headers) }
