# MetricSample.msg
# Custom metric sample from a MetricProbe plugin

string name
float64 value
string unit
string[] tag_keys
string[] tag_values
builtin_interfaces/Time timestamp
