# SystemAlert.msg
# Published when an anomaly or predicted failure is detected

std_msgs/Header header

string node_name
builtin_interfaces/Time timestamp

# Detection result
float32 anomaly_score           # [0.0 = normal, 1.0 = critical]
string severity                 # INFO | WARN | ERROR | CRITICAL | SCORE
string detector                 # isolation_forest | predictor/lstm | ...
string message

# Optional context
string[] affected_topics
string[] suggested_actions
