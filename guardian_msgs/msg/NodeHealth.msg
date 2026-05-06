# NodeHealth.msg
# Per-node health snapshot published at a configurable rate

std_msgs/Header header

string node_name
builtin_interfaces/Time timestamp

# Liveness
bool is_alive
string lifecycle_state          # unknown | inactive | active | error | ...

# Process metrics
float32 cpu_percent             # [0, 100]
float32 memory_mb               # RSS in MB
float32 memory_percent          # [0, 100]
uint32  thread_count
uint32  open_files

# Topic throughput
string[] published_topics
float32[] topic_rates_hz        # parallel array with published_topics

# Message latency
float32 avg_latency_ms
float32 max_latency_ms

# Custom metric samples (from MetricProbe plugins)
guardian_msgs/MetricSample[] custom_metrics
