TITLE FS0-only controlled excitatory conductance probe
COMMENT
Externally prescribed small conductance, used only to measure input transfer.
It is neither tACS nor an afferent/recurrent stochastic synapse implementation.
ENDCOMMENT
NEURON { POINT_PROCESS bs_synprobe_fs0 NONSPECIFIC_CURRENT i RANGE g, e, i }
UNITS { (mV) = (millivolt) (nA) = (nanoamp) (uS) = (microsiemens) }
PARAMETER { g = 0 (uS) e = 0 (mV) }
ASSIGNED { v (mV) i (nA) }
BREAKPOINT { i = g*(v-e) }
