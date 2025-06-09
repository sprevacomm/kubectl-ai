// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gollm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"k8s.io/klog/v2"
)

// Register the Claude provider factory on package initialization.
func init() {
	if err := RegisterProvider("claude", newClaudeClientFactory); err != nil {
		klog.Fatalf("Failed to register claude provider: %v", err)
	}
}

// newClaudeClientFactory is the factory function for creating Claude clients with options.
func newClaudeClientFactory(ctx context.Context, opts ClientOptions) (Client, error) {
	return NewClaudeClient(ctx, opts)
}

// ClaudeClient implements the gollm.Client interface for Anthropic's Claude models.
type ClaudeClient struct {
	client anthropic.Client
}

// Ensure ClaudeClient implements the Client interface.
var _ Client = &ClaudeClient{}

// NewClaudeClient creates a new client for interacting with Anthropic's Claude models.
// Supports custom HTTP client and skipVerifySSL via ClientOptions.
func NewClaudeClient(ctx context.Context, opts ClientOptions) (*ClaudeClient, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		return nil, errors.New("ANTHROPIC_API_KEY environment variable not set")
	}

	// Set up client options
	options := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	// Allow endpoint override via environment variable
	customEndpoint := os.Getenv("ANTHROPIC_API_URL")
	if customEndpoint != "" {
		options = append(options, option.WithBaseURL(customEndpoint))
		klog.Infof("Using custom Claude endpoint: %s", customEndpoint)
	}

	// Use custom HTTP client if SkipVerifySSL is set
	if opts.SkipVerifySSL {
		httpClient := createCustomHTTPClient(opts.SkipVerifySSL)
		options = append(options, option.WithHTTPClient(httpClient))
	}

	client := anthropic.NewClient(options...)

	return &ClaudeClient{
		client: client,
	}, nil
}

// Close cleans up any resources used by the client.
func (c *ClaudeClient) Close() error {
	// No specific cleanup needed for the Claude client currently.
	return nil
}

// StartChat starts a new chat session.
func (c *ClaudeClient) StartChat(systemPrompt, model string) Chat {
	// Default to Claude 4 Sonnet if no model is specified
	if model == "" {
		model = "claude-sonnet-4-0"
		klog.V(1).Info("No model specified, defaulting to claude-sonnet-4-0")
	}
	klog.V(1).Infof("Starting new Claude chat session with model: %s", model)

	// Initialize history - Claude doesn't use explicit system messages in history,
	// they are set as the system parameter in the API call
	return &claudeChatSession{
		client:       c.client,
		history:      []anthropic.MessageParam{},
		model:        model,
		systemPrompt: systemPrompt,
	}
}

// ClaudeCompletionResponse is a basic implementation of CompletionResponse.
type ClaudeCompletionResponse struct {
	content string
	usage   anthropic.Usage
}

// Response returns the completion content.
func (r *ClaudeCompletionResponse) Response() string {
	return r.content
}

// UsageMetadata returns the usage metadata.
func (r *ClaudeCompletionResponse) UsageMetadata() any {
	return r.usage
}

// GenerateCompletion sends a completion request to the Claude API.
func (c *ClaudeClient) GenerateCompletion(ctx context.Context, req *CompletionRequest) (CompletionResponse, error) {
	klog.Infof("Claude GenerateCompletion called with model: %s", req.Model)
	klog.V(1).Infof("Prompt:\n%s", req.Prompt)

	// Use the model specified in the request
	model := req.Model
	if model == "" {
		model = "claude-sonnet-4-0"
	}

	// Create the message request
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 1024,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(req.Prompt)),
		},
	}

	message, err := c.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to generate Claude completion: %w", err)
	}

	// Extract text content from the response
	if len(message.Content) == 0 {
		return nil, errors.New("received an empty response from Claude")
	}

	var content string
	for _, block := range message.Content {
		switch block := block.AsAny().(type) {
		case anthropic.TextBlock:
			content += block.Text
		}
	}

	if content == "" {
		return nil, errors.New("received a response from Claude with no text content")
	}

	resp := &ClaudeCompletionResponse{
		content: content,
		usage:   message.Usage,
	}

	return resp, nil
}

// SetResponseSchema is not implemented yet for Claude.
func (c *ClaudeClient) SetResponseSchema(schema *Schema) error {
	klog.Warning("ClaudeClient.SetResponseSchema is not implemented yet")
	return nil
}

// ListModels returns a list of available Claude models including Claude 4.
func (c *ClaudeClient) ListModels(ctx context.Context) ([]string, error) {
	// Claude models as of the current API including Claude 4
	return []string{
		// Claude 4 Models (latest and most capable)
		"claude-opus-4-0",
		"claude-sonnet-4-0",
		"claude-opus-4-20250514",
		"claude-sonnet-4-20250514",
		// Claude 3.7 Models
		"claude-3-7-sonnet-20250219",
		"claude-3-7-sonnet-latest",
		// Claude 3.5 Models
		"claude-3-5-sonnet-20241022",
		"claude-3-5-sonnet-latest",
		"claude-3-5-haiku-20241022",
		"claude-3-5-haiku-latest",
		"claude-3-5-sonnet-20240620", // previous version
		// Claude 3 Models
		"claude-3-opus-20240229",
		"claude-3-opus-latest",
		"claude-3-sonnet-20240229",
		"claude-3-haiku-20240307",
	}, nil
}

// --- Chat Session Implementation ---

type claudeChatSession struct {
	client              anthropic.Client
	history             []anthropic.MessageParam
	model               string
	systemPrompt        string
	functionDefinitions []*FunctionDefinition   // Stored in gollm format
	tools               []anthropic.ToolParam   // Stored in Claude format
}

// Ensure claudeChatSession implements the Chat interface.
var _ Chat = (*claudeChatSession)(nil)

// SetFunctionDefinitions stores the function definitions and converts them to Claude format.
func (cs *claudeChatSession) SetFunctionDefinitions(defs []*FunctionDefinition) error {
	cs.functionDefinitions = defs
	cs.tools = nil // Clear previous tools
	if len(defs) > 0 {
		cs.tools = make([]anthropic.ToolParam, len(defs))
		for i, gollmDef := range defs {
			// Convert gollm function definition to Claude tool format
			var params anthropic.ToolInputSchemaParam
			if gollmDef.Parameters != nil {
				// Convert the schema to Claude format
				bytes, err := gollmDef.Parameters.ToRawSchema()
				if err != nil {
					return fmt.Errorf("failed to convert schema for function %s: %w", gollmDef.Name, err)
				}
				
				// Parse the JSON schema
				var schemaMap map[string]interface{}
				if err := json.Unmarshal(bytes, &schemaMap); err != nil {
					return fmt.Errorf("failed to unmarshal schema for function %s: %w", gollmDef.Name, err)
				}

				// Extract properties if they exist
				if properties, ok := schemaMap["properties"]; ok {
					params.Properties = properties
				}
			}

			cs.tools[i] = anthropic.ToolParam{
				Name:        gollmDef.Name,
				Description: anthropic.String(gollmDef.Description),
				InputSchema: params,
			}
		}
	}
	klog.V(1).Infof("Set %d function definitions for Claude chat session", len(cs.functionDefinitions))
	return nil
}

// Send sends the user message(s), appends to history, and gets the LLM response.
func (cs *claudeChatSession) Send(ctx context.Context, contents ...any) (ChatResponse, error) {
	klog.V(1).Info("sending LLM request", "user", contents)

	// Convert contents to Claude message parts
	parts, err := cs.partsToClaudeBlocks(contents...)
	if err != nil {
		return nil, err
	}

	// Create user message
	userMessage := anthropic.MessageParam{
		Role:    anthropic.MessageParamRoleUser,
		Content: parts,
	}

	// Add to history
	cs.history = append(cs.history, userMessage)

	// Build the request parameters
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(cs.model),
		Messages:  cs.history,
		MaxTokens: 4096,
	}

	// Add system prompt if provided
	if cs.systemPrompt != "" {
		params.System = []anthropic.TextBlockParam{
			{Text: cs.systemPrompt},
		}
	}

	// Add tools if available - simplified for now
	if len(cs.tools) > 0 {
		// Convert to the expected format
		toolUnions := make([]anthropic.ToolUnionParam, len(cs.tools))
		for i, tool := range cs.tools {
			toolUnions[i] = anthropic.ToolUnionParam{
				OfTool: &tool,
			}
		}
		params.Tools = toolUnions
	}

	// Send the request
	message, err := cs.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to send message to Claude: %w", err)
	}

	// Add Claude's response to history
	cs.history = append(cs.history, message.ToParam())

	klog.V(1).Info("got LLM response", "response", message)
	return &ClaudeChatResponse{claudeMessage: message}, nil
}

// SendStreaming sends a streaming request (not implemented yet for Claude)
func (cs *claudeChatSession) SendStreaming(ctx context.Context, contents ...any) (ChatResponseIterator, error) {
	// For now, fallback to non-streaming
	response, err := cs.Send(ctx, contents...)
	if err != nil {
		return nil, err
	}

	// Return a simple iterator that yields the response once
	return func(yield func(ChatResponse, error) bool) {
		yield(response, nil)
	}, nil
}

// IsRetryableError determines if an error from the Claude API should be retried.
func (cs *claudeChatSession) IsRetryableError(err error) bool {
	if err == nil {
		return false
	}
	return DefaultIsRetryableError(err)
}

// partsToClaudeBlocks converts gollm content to Claude content blocks
func (cs *claudeChatSession) partsToClaudeBlocks(contents ...any) ([]anthropic.ContentBlockParamUnion, error) {
	var parts []anthropic.ContentBlockParamUnion

	for _, content := range contents {
		switch v := content.(type) {
		case string:
			parts = append(parts, anthropic.NewTextBlock(v))
		case FunctionCallResult:
			// Convert the result to string
			var resultStr string
			if v.Result != nil {
				// Marshal the map to JSON
				resultBytes, err := json.Marshal(v.Result)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal function result: %w", err)
				}
				resultStr = string(resultBytes)
			}
			// Use the proper tool result block
			parts = append(parts, anthropic.NewToolResultBlock(v.ID, resultStr, false))
		default:
			return nil, fmt.Errorf("unexpected type of content: %T", content)
		}
	}
	return parts, nil
}

// --- Helper structs for ChatResponse interface ---

type ClaudeChatResponse struct {
	claudeMessage *anthropic.Message
}

var _ ChatResponse = (*ClaudeChatResponse)(nil)

func (r *ClaudeChatResponse) UsageMetadata() any {
	if r.claudeMessage != nil {
		return r.claudeMessage.Usage
	}
	return nil
}

func (r *ClaudeChatResponse) Candidates() []Candidate {
	if r.claudeMessage == nil {
		return nil
	}
	// Claude doesn't have multiple candidates like some other models,
	// so we return a single candidate with the message content
	return []Candidate{&ClaudeCandidate{claudeMessage: r.claudeMessage}}
}

type ClaudeCandidate struct {
	claudeMessage *anthropic.Message
}

var _ Candidate = (*ClaudeCandidate)(nil)

func (c *ClaudeCandidate) Parts() []Part {
	if c.claudeMessage == nil {
		return nil
	}

	var parts []Part
	for _, content := range c.claudeMessage.Content {
		switch block := content.AsAny().(type) {
		case anthropic.TextBlock:
			parts = append(parts, &ClaudePart{textContent: block.Text})
		case anthropic.ToolUseBlock:
			parts = append(parts, &ClaudePart{toolUse: &block})
		}
	}
	return parts
}

// String provides a simple string representation for logging/debugging.
func (c *ClaudeCandidate) String() string {
	if c.claudeMessage == nil {
		return "<nil candidate>"
	}
	var content string
	for _, block := range c.claudeMessage.Content {
		switch b := block.AsAny().(type) {
		case anthropic.TextBlock:
			content += b.Text
		case anthropic.ToolUseBlock:
			content += fmt.Sprintf("[Tool Use: %s]", b.Name)
		}
	}
	return fmt.Sprintf("ClaudeCandidate(Content: %q)", content)
}

type ClaudePart struct {
	textContent string
	toolUse     *anthropic.ToolUseBlock
}

var _ Part = (*ClaudePart)(nil)

func (p *ClaudePart) AsText() (string, bool) {
	return p.textContent, p.textContent != ""
}

func (p *ClaudePart) AsFunctionCalls() ([]FunctionCall, bool) {
	if p.toolUse == nil {
		return nil, false
	}

	// Convert Claude tool use to gollm function call
	var args map[string]any
	if p.toolUse.Input != nil {
		// Parse the JSON input
		if err := json.Unmarshal(p.toolUse.Input, &args); err != nil {
			klog.Warningf("Failed to unmarshal tool input: %v", err)
			args = make(map[string]any)
		}
	}

	gollmCalls := []FunctionCall{
		{
			ID:        p.toolUse.ID,
			Name:      p.toolUse.Name,
			Arguments: args,
		},
	}
	return gollmCalls, true
} 