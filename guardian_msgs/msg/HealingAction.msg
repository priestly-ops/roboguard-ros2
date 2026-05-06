# HealingAction.msg
# Audit log entry for every healing action executed

std_msgs/Header header

string node_name
builtin_interfaces/Time timestamp

string action_name              # soft_restart | hard_restart | parameter_reset | ...
string status                   # STARTED | SUCCEEDED | FAILED | ESCALATED
string message
float32 duration_seconds
uint32  attempt_number
