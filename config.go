package main

import (
	"fmt"
	"net/url"
	"os"
	"strings"
)

const (
	envAPIKey   = "OPENAI_API_KEY"
	envEndpoint = "OPENAI_ENDPOINT"
	envBaseURL  = "OPENAI_BASE_URL"
	envModel    = "OPENAI_MODEL"
)

type Config struct {
	APIKey   string
	Endpoint string
	Model    string
}

func loadConfig() (Config, error) {
	apiKey := strings.TrimSpace(os.Getenv(envAPIKey))
	exactEndpoint := strings.TrimSpace(os.Getenv(envEndpoint))
	baseURL := strings.TrimSpace(os.Getenv(envBaseURL))
	model := strings.TrimSpace(os.Getenv(envModel))

	var missing []string
	if apiKey == "" {
		missing = append(missing, envAPIKey)
	}
	if exactEndpoint == "" && baseURL == "" {
		missing = append(missing, envEndpoint+" (or "+envBaseURL+")")
	}
	if model == "" {
		missing = append(missing, envModel)
	}
	if len(missing) > 0 {
		return Config{}, fmt.Errorf("missing required environment variables: %s", strings.Join(missing, ", "))
	}

	endpoint := exactEndpoint
	var err error
	if endpoint != "" {
		endpoint, err = validateEndpoint(endpoint)
	} else {
		endpoint, err = chatCompletionsEndpoint(baseURL)
	}
	if err != nil {
		return Config{}, fmt.Errorf("invalid OpenAI endpoint: %w", err)
	}

	return Config{APIKey: apiKey, Endpoint: endpoint, Model: model}, nil
}

func validateEndpoint(value string) (string, error) {
	u, err := parseEndpointURL(value)
	if err != nil {
		return "", err
	}
	return u.String(), nil
}

func chatCompletionsEndpoint(base string) (string, error) {
	u, err := parseEndpointURL(base)
	if err != nil {
		return "", err
	}

	path := strings.TrimRight(u.Path, "/")
	switch {
	case strings.HasSuffix(path, "/chat/completions"):
		u.Path = path
	case strings.HasSuffix(path, "/v1"):
		u.Path = path + "/chat/completions"
	case path == "":
		u.Path = "/v1/chat/completions"
	default:
		u.Path = path + "/v1/chat/completions"
	}
	return u.String(), nil
}

func parseEndpointURL(value string) (*url.URL, error) {
	u, err := url.Parse(strings.TrimSpace(value))
	if err != nil {
		return nil, err
	}
	if (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, fmt.Errorf("must be a URL with an http(s) scheme and host")
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("must not contain a query or fragment")
	}
	return u, nil
}
