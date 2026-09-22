TITLE FS0-only equilibrium rectifying tonic conductance
COMMENT
Voltage dependence inspired by L23Net tonic.mod (Pavlov et al. 2009).
This is an explicitly reduced equilibrium I--V relation, not a transplanted
mammalian channel kinetic model. No Q10 rescaling. Stable analytic rate limits
replace the source mechanism's near-zero special cases.
ENDCOMMENT
NEURON {
    SUFFIX bs_tonic_fs0
    NONSPECIFIC_CURRENT i
    RANGE gbar, e, i, open_fraction
}
UNITS { (mV) = (millivolt) (mA) = (milliamp) (S) = (siemens) }
PARAMETER { gbar = 0 (S/cm2) e = -75 (mV) }
ASSIGNED { v (mV) i (mA/cm2) open_fraction }
BREAKPOINT {
    open_fraction = activation(v)
    i = gbar * open_fraction * (v-e)
}
FUNCTION exprel_stable(x) {
    if (fabs(x) < 1e-4) {
        exprel_stable = 1 + x/2 + x*x/12
    } else {
        exprel_stable = x/(1-exp(-x))
    }
}
FUNCTION activation(vm (mV)) {
    LOCAL a, b
    a = 50 * exprel_stable(0.1*(vm+20))
    b = 20 * exprel_stable(-0.08*(vm-10))
    activation = a/(a+b)
}
