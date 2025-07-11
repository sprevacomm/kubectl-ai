# kubectl-ai

> **Note:** This is a fork of [kubectl-ai](https://github.com/GoogleCloudPlatform/kubectl-ai) that adds comprehensive Claude AI integration. The original project supports multiple providers, and this fork extends that support with enhanced Claude functionality, timeout configuration, and cost-effective model options.

[![Go Report Card](https://goreportcard.com/badge/github.com/GoogleCloudPlatform/kubectl-ai)](https://goreportcard.com/report/github.com/GoogleCloudPlatform/kubectl-ai)
![GitHub License](https://img.shields.io/github/license/GoogleCloudPlatform/kubectl-ai)
[![GitHub stars](https://img.shields.io/github/stars/GoogleCloudPlatform/kubectl-ai.svg)](https://github.com/GoogleCloudPlatform/kubectl-ai/stargazers)

`kubectl-ai` acts as an intelligent interface, translating user intent into
precise Kubernetes operations, making Kubernetes management more accessible and
efficient.

![kubectl-ai demo GIF using: kubectl-ai "how's nginx app doing in my cluster"](./.github/kubectl-ai.gif)

## Quick Start

First, ensure that kubectl is installed and configured.

### Installation

#### Quick Install (Linux & MacOS only)

```shell
curl -sSL https://raw.githubusercontent.com/GoogleCloudPlatform/kubectl-ai/main/install.sh | bash
```

<details>

<summary>Other Installation Methods</summary>

#### Manual Installation (Linux, MacOS and Windows)

1. Download the latest release from the [releases page](https://github.com/GoogleCloudPlatform/kubectl-ai/releases/latest) for your target machine.

2. Untar the release, make the binary executable and move it to a directory in your $PATH (as shown below).

```shell
tar -zxvf kubectl-ai_Darwin_arm64.tar.gz
chmod a+x kubectl-ai
sudo mv kubectl-ai /usr/local/bin/
```

#### Install with Krew (Linux/macOS/Windows)
First of all, you need to have krew insatlled, refer to [krew document](https://krew.sigs.k8s.io/docs/user-guide/setup/install/) for more details
Then you can install with krew
```shell
kubectl krew install ai
```
Now you can invoke `kubectl-ai` as a kubectl plugin like this: `kubectl ai`.

#### Building from Source

If you want to build kubectl-ai from source or install your modified version:

```bash
# Clone this repository
git clone https://github.com/your-username/kubectl-ai.git
cd kubectl-ai

# Build the binary
go build -o kubectl-ai ./cmd

# Install to system path (requires sudo)
sudo cp kubectl-ai /usr/local/bin/kubectl-ai

# Or install to user bin (no sudo required, but make sure ~/bin is in your PATH)
mkdir -p ~/bin
cp kubectl-ai ~/bin/kubectl-ai

# Verify installation
kubectl-ai version
```

**Note:** After making changes to the code, repeat the build and install steps to use your updated version:

```bash
# Rebuild and reinstall after changes
go build -o kubectl-ai ./cmd
sudo cp kubectl-ai /usr/local/bin/kubectl-ai
```
</details>

### Usage

`kubectl-ai` supports AI models from `gemini`, `vertexai`, `azopenai`, `openai`, `grok`, `claude` and local LLM providers such as `ollama` and `llama.cpp`.

#### Using Gemini (Default)

Set your Gemini API key as an environment variable. If you don't have a key, get one from [Google AI Studio](https://aistudio.google.com).

```bash
export GEMINI_API_KEY=your_api_key_here
kubectl-ai

# Use different gemini model
kubectl-ai --model gemini-2.5-pro-exp-03-25

# Use 2.5 flash (faster) model
kubectl-ai --quiet --model gemini-2.5-flash-preview-04-17 "check logs for nginx app in hello namespace"
```

<details>

<summary>Use other AI models</summary>

#### Using AI models running locally (ollama or llama.cpp)

You can use `kubectl-ai` with AI models running locally. `kubectl-ai` supports [ollama](https://ollama.com/) and [llama.cpp](https://github.com/ggml-org/llama.cpp) to use the AI models running locally.

Additionally, the [`modelserving`](modelserving/) directory provides tools and instructions for deploying your own `llama.cpp`-based LLM serving endpoints locally or on a Kubernetes cluster. This allows you to host models like Gemma directly in your environment.

An example of using Google's `gemma3` model with `ollama`:

```shell
# assuming ollama is already running and you have pulled one of the gemma models
# ollama pull gemma3:12b-it-qat

# if your ollama server is at remote, use OLLAMA_HOST variable to specify the host
# export OLLAMA_HOST=http://192.168.1.3:11434/

# enable-tool-use-shim because models require special prompting to enable tool calling
kubectl-ai --llm-provider ollama --model gemma3:12b-it-qat --enable-tool-use-shim

# you can use `models` command to discover the locally available models
>> models
```

#### Using Grok

You can use X.AI's Grok model by setting your X.AI API key:

```bash
export GROK_API_KEY=your_xai_api_key_here
kubectl-ai --llm-provider=grok --model=grok-3-beta
```

#### Using Claude

You can use Anthropic's Claude models by setting your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
kubectl-ai --llm-provider=claude --model=claude-sonnet-4-0

# Use Claude 4 Opus (most capable) model
kubectl-ai --llm-provider=claude --model=claude-opus-4-0 "debug the failing deployment"

# Use Claude 4 Sonnet (balanced performance and capability) 
kubectl-ai --llm-provider=claude --model=claude-sonnet-4-0 "analyze the resource usage"

# Use Claude 3.5 Haiku (faster) model for quick queries
kubectl-ai --quiet --llm-provider=claude --model=claude-3-5-haiku-20241022 "check logs for nginx app in hello namespace"
```

#### Using Azure OpenAI

You can also use Azure OpenAI deployment by setting your OpenAI API key and specifying the provider:

```bash
export AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
export AZURE_OPENAI_ENDPOINT=https://your_azure_openai_endpoint_here
kubectl-ai --llm-provider=azopenai --model=your_azure_openai_deployment_name_here
# or
az login
kubectl-ai --llm-provider=openai://your_azure_openai_endpoint_here --model=your_azure_openai_deployment_name_here
```

#### Using OpenAI

You can also use OpenAI models by setting your OpenAI API key and specifying the provider:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
kubectl-ai --llm-provider=openai --model=gpt-4.1
```

#### Using OpenAI Compatible API
For example, you can use aliyun qwen-xxx models as follows
```bash
export OPENAI_API_KEY=your_openai_api_key_here
export OPENAI_ENDPOINT=https://dashscope.aliyuncs.com/compatible-mode/v1
kubectl-ai --llm-provider=openai --model=qwen-plus
```
</details>

Run interactively:

```shell
kubectl-ai
```

The interactive mode allows you to have a chat with `kubectl-ai`, asking multiple questions in sequence while maintaining context from previous interactions. Simply type your queries and press Enter to receive responses. To exit the interactive shell, type `exit` or press Ctrl+C.

Or, run with a task as input:

```shell
kubectl-ai --quiet "fetch logs for nginx app in hello namespace"
```

Combine it with other unix commands:

```shell
kubectl-ai < query.txt
# OR
echo "list pods in the default namespace" | kubectl-ai
```

You can even combine a positional argument with stdin input. The positional argument will be used as a prefix to the stdin content:

```shell
cat error.log | kubectl-ai "explain the error"
```

## Claude Integration Quick Start Guide

This guide will walk you through testing the new Claude AI integration step by step.

### Prerequisites

1. **kubectl installed and configured** with access to a Kubernetes cluster
2. **Go 1.24+ installed** (if building from source)
3. **Anthropic API key** - Get one from [Anthropic Console](https://console.anthropic.com/)

### Step 1: Get Your Anthropic API Key

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-ant-`)

### Step 2: Set Environment Variables

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-your_api_key_here

# Verify it's set
echo $ANTHROPIC_API_KEY
```

### Step 3: Build kubectl-ai (if needed)

If you haven't installed kubectl-ai yet or want the latest version with Claude support:

```bash
# Clone the repository
git clone https://github.com/GoogleCloudPlatform/kubectl-ai.git
cd kubectl-ai

# Build the binary
go build -o kubectl-ai ./cmd
```

### Step 4: Test Claude Integration

**4.1 Test that Claude is recognized:**
```bash
./kubectl-ai --llm-provider=claude --help
```

**4.2 List available Claude models:**
```bash
./kubectl-ai --llm-provider=claude --quiet "list available models"
```

**4.3 Test with Claude 4 Sonnet (balanced performance):**
```bash
./kubectl-ai --llm-provider=claude --model=claude-sonnet-4-0 "show me all pods in the default namespace"
```

**4.4 Test with Claude 4 Opus (most capable):**
```bash
./kubectl-ai --llm-provider=claude --model=claude-opus-4-0 "explain what a deployment is in Kubernetes"
```

**4.5 Test with Claude 3.5 Haiku (fastest):**
```bash
./kubectl-ai --llm-provider=claude --model=claude-3-5-haiku-20241022 --quiet "get cluster info"
```

### Step 5: Interactive Mode Testing

Start an interactive session with Claude:

```bash
./kubectl-ai --llm-provider=claude --model=claude-sonnet-4-0
```

Try these commands in the interactive mode:
```
> list all namespaces
> show me pods in kube-system namespace
> explain the difference between a service and an ingress
> exit
```

### Step 6: Verify Available Models

Create a simple test script to see all Claude models:

```bash
# Quick model list test
./kubectl-ai --llm-provider=claude --quiet "what models are available?" | grep -i claude
```

### Expected Claude Models Available:

- **Claude 4**: `claude-opus-4-0`, `claude-sonnet-4-0`
- **Claude 3.7**: `claude-3-7-sonnet-20250219`, `claude-3-7-sonnet-latest`
- **Claude 3.5**: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
- **Claude 3**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`

### Step 7: Configure Timeouts (Optional)

If you experience timeout issues with long analysis queries, you can increase the Claude API timeout:

```bash
# Set Claude API response timeout (default: 60 seconds)
export CLAUDE_API_TIMEOUT=180s  # 3 minutes

# Set overall request timeout (default: 300 seconds)
export CLAUDE_REQUEST_TIMEOUT=600s  # 10 minutes

# Test with timeout configuration
./kubectl-ai --llm-provider=claude --model=claude-3-5-haiku-20241022 "comprehensive cluster analysis"
```

**Available timeout formats:**
- Duration strings: `60s`, `5m`, `1h30m`
- Seconds as integers: `180` (equivalent to 180 seconds)

### Troubleshooting

**If you get "ANTHROPIC_API_KEY environment variable not set":**
```bash
# Make sure your API key is properly exported
export ANTHROPIC_API_KEY=sk-ant-your_actual_key_here
```

**If you get "unknown provider: claude":**
```bash
# Rebuild kubectl-ai to ensure Claude provider is included
go build -o kubectl-ai ./cmd
```

**If you get build errors:**
```bash
# Ensure you have Go 1.24+ and run
go mod tidy
go build -o kubectl-ai ./cmd
```

**If you get "context deadline exceeded" with Claude:**
```bash
# Increase timeout for long analysis queries
export CLAUDE_API_TIMEOUT=300s      # 5 minutes for API response
export CLAUDE_REQUEST_TIMEOUT=600s  # 10 minutes total

# Or break down complex queries into smaller parts
kubectl-ai --llm-provider=claude --model=claude-3-5-haiku-20241022 "show pod distribution by node"
kubectl-ai --llm-provider=claude --model=claude-3-5-haiku-20241022 "check scheduling policies"
```

## Claude Prompt Examples & Best Practices

Get the most out of every Claude API call with these proven prompting strategies:

### Effective Query Patterns

#### 1. Comprehensive Analysis in One Call
Instead of multiple small queries, use structured prompts that gather everything at once:

```bash
# GOOD: Comprehensive but focused
kai "Analyze the 'harness' namespace: show pod distribution across nodes, check for any scheduling constraints or affinities, identify resource bottlenecks, and summarize any issues found"

# AVOID: Multiple separate calls
kai "show pods in harness namespace"
kai "check node distribution" 
kai "look for scheduling policies"
```

#### 2. Troubleshooting with Context
Provide clear context about what you're investigating:

```bash
# EXCELLENT: Context + specific request
kai "We're seeing uneven pod distribution in our cluster. Please analyze the harness namespace and determine if worker1 is being favored for scheduling. Include: pod counts per node, any node selectors or affinities, recent scheduling events, and resource utilization differences."

# GOOD: Specific troubleshooting
kai "Debug why pods in the 'app=frontend' deployment are failing to start. Check events, resource limits, image pull status, and node capacity."
```

#### 3. Multi-Step Operations
Break complex tasks into logical steps within one prompt:

```bash
# EXCELLENT: Structured workflow
kai "Help me safely scale down the production workload: 1) Check current pod distribution and resource usage, 2) Identify which pods can be safely removed, 3) Show me the kubectl commands to scale down gradually, 4) Explain what to monitor during the process."
```

### Specific Analysis Examples

#### Resource Investigation
```bash
# Memory and CPU analysis
kai "Analyze resource consumption across all namespaces. Show top consumers, identify pods near limits, check for any resource pressure on nodes, and recommend optimization opportunities."

# Storage analysis  
kai "Investigate persistent volume usage: show PV/PVC status, identify unused volumes, check for storage class issues, and highlight any capacity concerns."
```

#### Security & Configuration
```bash
# Security audit
kai "Perform a security review of the 'production' namespace: check RBAC permissions, identify pods running as root, review network policies, and flag any security misconfigurations."

# Configuration review
kai "Review deployment configurations in 'backend' namespace for best practices: check resource limits, health checks, update strategies, and security contexts."
```

#### Performance & Scaling
```bash
# Performance analysis
kai "Analyze application performance metrics: check HPA status, review recent scaling events, identify bottlenecks, and suggest optimization strategies for better resource utilization."

# Capacity planning
kai "Help with capacity planning: analyze current resource usage trends, identify nodes approaching capacity, calculate headroom for growth, and recommend scaling strategies."
```

### Efficiency Tips

#### 1. Use Specific Kubernetes Language
```bash
# EXCELLENT: Uses K8s terminology
kai "Check if any DaemonSets are missing replicas, review StatefulSet rolling update status, and verify all Services have healthy endpoints."

# GOOD: Specific resource types
kai "Analyze all PodDisruptionBudgets and show which deployments might be affected during node maintenance."
```

#### 2. Request Actionable Output
```bash
# EXCELLENT: Asks for specific actions
kai "Identify all failing pods and provide the exact kubectl commands needed to restart them, including any dependencies that should be restarted first."

# GOOD: Requests explanation with commands
kai "Show me how to troubleshoot this CrashLoopBackOff issue and provide step-by-step debugging commands."
```

#### 3. Combine Analysis with Solutions
```bash
# EXCELLENT: Problem + solution in one call
kai "Our ingress is returning 503 errors. Diagnose the issue by checking ingress configuration, backend services, pod readiness, and provide specific fix commands."

# GOOD: Analysis with recommendations
kai "Review our cluster's resource allocation and suggest specific requests/limits adjustments for better efficiency."
```

### Advanced Usage Patterns

#### Monitoring & Alerting
```bash
# Health check
kai "Perform a comprehensive cluster health check: verify all system components, check for any failing pods across all namespaces, review recent events for issues, and summarize overall cluster status."

# Event analysis
kai "Analyze the last 2 hours of cluster events to identify patterns, recurring issues, or anomalies that need attention."
```

#### Migration & Updates
```bash
# Update planning
kai "Plan a rolling update for 'api-service' deployment: check current status, verify update strategy, identify potential issues, and provide the safest update sequence."

# Migration assistance  
kai "Help migrate workloads from 'old-namespace' to 'new-namespace': show current resources, generate migration manifests, and provide validation steps."
```

### Avoiding Timeouts

#### Break Down Complex Queries
Instead of one massive analysis, use focused queries:

```bash
# If this times out:
kai "Complete infrastructure audit of entire cluster..."

# Try this instead:
kai "Audit compute resources: analyze node utilization, pod distribution, and resource constraints"
kai "Audit networking: review services, ingresses, network policies, and connectivity issues"  
kai "Audit storage: check PV/PVC status, storage classes, and capacity planning"
```

#### Use Incremental Analysis
```bash
# Start broad, then drill down:
kai "Overview of cluster health and identify top 3 areas needing attention"
# Then follow up with specific areas identified
```

### Pro Tips for Maximum Effectiveness

1. Be Specific About Output Format
   ```bash
   kai "Show pod resource usage in table format with node assignments"
   kai "Generate YAML manifests for the recommended changes"
   ```

2. Include Context About Your Environment
   ```bash
   kai "In our production EKS cluster, analyze why the payment service pods are experiencing high latency"
   ```

3. Ask for Explanations
   ```bash
   kai "Explain why these pods are in Pending state and show me how to fix each root cause"
   ```

4. Request Validation Steps
   ```bash
   kai "Help me scale this deployment and provide validation commands to confirm the scaling worked correctly"
   ```

### Success!

If all steps work correctly, you now have kubectl-ai with full Claude integration including Claude 4 support! 

The integration supports:
- All Claude models (including Claude 4)
- Function calling for kubectl operations
- Interactive and non-interactive modes
- Streaming responses
- Tool usage for Kubernetes operations

---

## Tools

`kubectl-ai` leverages LLMs to suggest and execute Kubernetes operations using a set of powerful tools. It comes with built-in tools like `kubectl` and `bash`.

You can also extend its capabilities by defining your own custom tools. By default, `kubectl-ai` looks for your tool configurations in `~/.config/kubectl-ai/tools.yaml`.

To specify tools configuration files or directories containing tools configuration files, use:

```shell
kubectl-ai --custom-tools-config=YOUR_CONFIG
```

You can include multiple tools in a single configuration file, or a directory with multiple configuration files, each dedicated to a single or multiple tools.
Define your custom tools using the following schema:

```yaml
- name: tool_name
  description: "A clear description that helps the LLM understand when to use this tool."
  command: "your_command" # For example: 'gcloud' or 'gcloud container clusters'
  command_desc: "Detailed information for the LLM, including command syntax and usage examples."
```

A custom tool definition for `helm` could look like the following example:

```yaml
- name: helm
  description: "Helm is the Kubernetes package manager and deployment tool. Use it to define, install, upgrade, and roll back applications packaged as Helm charts in a Kubernetes cluster."
  command: "helm"
  command_desc: |
    Helm command-line interface, with the following core subcommands and usage patterns:    
    - helm install <release-name> <chart> [flags]  
      Install a chart into the cluster.      
    - helm upgrade <release-name> <chart> [flags]  
      Upgrade an existing release to a new chart version or configuration.      
    - helm list [flags]  
      List all releases in one or all namespaces.      
    - helm uninstall <release-name> [flags]  
      Uninstall a release and clean up associated resources.  
    Use `helm --help` or `helm <subcommand> --help` to see full syntax, available flags, and examples for each command.
```

## MCP Client Mode

> **Note:** MCP Client Mode is available in `kubectl-ai` version v0.0.12 and onwards.

`kubectl-ai` can connect to external [MCP](https://modelcontextprotocol.io/examples) Servers to access additional tools in addition to built-in tools.

### Quick Start

Enable MCP client mode:

```bash
kubectl-ai --mcp-client
```

### Configuration

Create or edit `~/.config/kubectl-ai/mcp.yaml` to customize MCP servers:

```yaml
servers:
  # Local MCP server (stdio-based)
  # sequential-thinking: Advanced reasoning and step-by-step analysis
  - name: sequential-thinking
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-sequential-thinking"
  
  # Remote MCP server (HTTP-based)
  - name: cloudflare-documentation
    url: https://docs.mcp.cloudflare.com/mcp
    
  # Optional: Remote MCP server with authentication
  - name: custom-api
    url: https://api.example.com/mcp
    auth:
      type: "bearer"
      token: "${MCP_TOKEN}"
```

The system automatically:
- Converts parameter names (snake_case → camelCase)
- Handles type conversion (strings → numbers/booleans when appropriate)
- Provides fallback behavior for unknown servers

No additional setup required - just use the `--mcp-client` flag and the AI will have access to all configured MCP tools.

📖 **For detailed configuration options, troubleshooting, and advanced features for MCP Client mode, see the [MCP Client Documentation](pkg/mcp/README.md).**

## Extras

You can use the following special keywords for specific actions:

* `model`: Display the currently selected model.
* `models`: List all available models.
* `tools`: List all available tools.
* `version`: Display the `kubectl-ai` version.
* `reset`: Clear the conversational context.
* `clear`: Clear the terminal screen.
* `exit` or `quit`: Terminate the interactive shell (Ctrl+C also works).

### Invoking as kubectl plugin

You can also run `kubectl ai`. `kubectl` finds any executable file in your `PATH` whose name begins with `kubectl-` as a [plugin](https://kubernetes.io/docs/tasks/extend-kubectl/kubectl-plugins/).

## MCP Server Mode

`kubectl-ai` can also act as an MCP server that exposes `kubectl` as a tool for other MCP clients (like Claude, Cursor, or VS Code) to interact with your locally configured Kubernetes environment. 

Enable MCP server mode:

```bash
kubectl-ai --mcp-server
```

This allows AI agents and tools to execute kubectl commands in your environment through the Model Context Protocol.

📖 **For details on configuring kubectl-ai as an MCP server for use with Claude, Cursor, VS Code, and other MCP clients, see the [MCP Server Documentation](./docs/mcp.md).**

## k8s-bench

kubectl-ai project includes [k8s-bench](./k8s-bench/README.md) - a benchmark to evaluate performance of different LLM models on kubernetes related tasks. Here is a summary from our last run:

| Model | Success | Fail |
|-------|---------|------|
| gemini-2.5-flash-preview-04-17 | 10 | 0 |
| gemini-2.5-pro-preview-03-25 | 10 | 0 |
| gemma-3-27b-it | 8 | 2 |
| **Total** | 28 | 2 |

See [full report](./k8s-bench.md) for more details.

## Start Contributing

We welcome contributions to `kubectl-ai` from the community. Take a look at our
[contribution guide](contributing.md) to get started.

---

*Note: This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).*
